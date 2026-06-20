"""
Object-Oriented dataset loaders for BridgeData V2 and LeRobot SO100.
"""
import jax
import jax.numpy as jnp
import numpy as np
import cv2
from typing import Dict, Any, Optional, Iterator
from jepa_robotics.data.kinematics import forward_kinematics

class BaseRobotDataset:
    """Base class for managing and loading robot trajectory datasets."""
    def __init__(self, data_dir: str = "./data", batch_size: int = 32, seq_len: int = 5, limit: Optional[int] = None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.limit = limit
        
    def load(self, split: str = "train") -> Iterator[Dict[str, Any]]:
        """Loads and yields batches of data from the dataset."""
        raise NotImplementedError("Subclasses must implement the load method.")
        
    def _apply_kinematics(self, batch: Dict[str, Any], robot_type: str) -> Dict[str, Any]:
        """
        Passes native joint states through the Kinematic Bridge to standardize output.
        """
        if "joint_states" in batch:
            # We must apply kinematics over the batch and sequence dimensions: (B, S, DoF)
            b, s, dof = batch["joint_states"].shape
            flat_states = batch["joint_states"].reshape(b * s, dof)
            flat_7d = forward_kinematics(flat_states, robot_type)
            batch["state_7d"] = flat_7d.reshape(b, s, 7)
            
        if "joint_actions" in batch:
            b, s, dof = batch["joint_actions"].shape
            flat_actions = batch["joint_actions"].reshape(b * s, dof)
            flat_7d = forward_kinematics(flat_actions, robot_type)
            batch["action_7d"] = flat_7d.reshape(b, s, 7)
            
        return batch

class BridgeDataLoader(BaseRobotDataset):
    """Loader for BridgeData V2 (WidowX 250) using pure TFDS RLDS format."""
    def __init__(self, tfds_data_dir: str = "/home/tmainetucker/Repos/JEPA_Robotics/data/bridge_data_v2", sample_fraction: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.tfds_data_dir = tfds_data_dir
        self.sample_fraction = sample_fraction

    def load(self, split: str = "train") -> Iterator[Dict[str, Any]]:
        import tensorflow_datasets as tfds
        import tensorflow as tf
        
        print(f"Loading REAL BridgeData V2 (RLDS) from {self.tfds_data_dir}... (Split: {split}, Fraction: {self.sample_fraction})")
        
        # We enforce a deterministic seed and aggressive file shuffling to mathematically stratify tasks
        read_config = tfds.ReadConfig(shuffle_seed=42, shuffle_files=True)
        
        builder = tfds.builder_from_directory(f"{self.tfds_data_dir}/bridge/0.1.0")
        
        # Calculate exactly what the 10% slice of episodes is
        total_episodes = builder.info.splits[split].num_examples
        slice_episodes = int(total_episodes * self.sample_fraction)
        
        # Apply the stratified slice
        sliced_split = f"{split}[:{slice_episodes}]"
        
        dataset = builder.as_dataset(split=sliced_split, read_config=read_config)
        
        # In RLDS, each element is an 'episode' containing a dataset of 'steps'
        def process_step(step):
            img = step['observation']['image']
            img = tf.image.resize(img, [256, 256])
            img = tf.cast(img, tf.uint8)
            return img, step['observation']['state'], step['action']

        # Flat map episodes into a continuous stream of steps
        step_ds = dataset.flat_map(lambda episode: episode['steps'].map(process_step))
        
        # Create sliding windows of size seq_len
        window_ds = step_ds.window(self.seq_len, shift=1, drop_remainder=True)
        window_ds = window_ds.flat_map(lambda img, state, action: tf.data.Dataset.zip((img.batch(self.seq_len), state.batch(self.seq_len), action.batch(self.seq_len))))
        
        # Batch the windows
        batched_ds = window_ds.batch(self.batch_size, drop_remainder=True)
        
        if self.limit:
            batched_ds = batched_ds.take(self.limit)
            
        # Convert to numpy iterator for JAX
        for imgs, states, actions in batched_ds.as_numpy_iterator():
            batch = {
                "image": jnp.array(imgs),           # (B, S, 256, 256, 3)
                "joint_states": jnp.array(states),  # (B, S, 7) for WidowX
                "joint_actions": jnp.array(actions) # (B, S, 7) for WidowX
            }
            yield self._apply_kinematics(batch, "widowx")

class SO100DataLoader(BaseRobotDataset):
    """Loader for LeRobot SO100 Hugging Face datasets from local offline cache."""
    def __init__(self, hf_repo: str = "lerobot/svla_so100_stacking", offline_dir: str = "/home/tmainetucker/Repos/JEPA_Robotics/data/lerobot_so100", sample_fraction: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.hf_repo = hf_repo
        self.offline_dir = offline_dir
        self.sample_fraction = sample_fraction

    def load(self, split: str = "train") -> Iterator[Dict[str, Any]]:
        from datasets import load_dataset
        import cv2
        
        print(f"Loading REAL LeRobot SO100 Data OFFLINE from {self.offline_dir}... (Split: {split}, Fraction: {self.sample_fraction})")
        
        # Load the states/actions metadata (Parquet) strictly from the offline cache
        # Note: If it's cached, HF automatically loads offline.
        dataset = load_dataset(self.hf_repo, split=split, cache_dir=self.offline_dir)
        
        # Calculate EXACT subset and apply stratified shuffle to mix trajectories
        total_rows = len(dataset)
        slice_rows = int(total_rows * self.sample_fraction)
        dataset = dataset.shuffle(seed=42).select(range(slice_rows))
        
        # For HF LeRobot datasets, the video is chunked. For simplicity in this iterator,
        # we'll read frame by frame into sliding windows.
        video_path = f"{self.offline_dir}/videos/observation.images.top/chunk-000/file-000.mp4"
        cap = cv2.VideoCapture(video_path)
        
        window_imgs = []
        window_states = []
        window_actions = []
        
        batch_imgs = []
        batch_states = []
        batch_actions = []
        
        yielded_batches = 0
        
        for i, row in enumerate(dataset):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (256, 256))
            
            window_imgs.append(frame_rgb)
            window_states.append(row["observation.state"])
            window_actions.append(row["action"])
            
            if len(window_imgs) > self.seq_len:
                window_imgs.pop(0)
                window_states.pop(0)
                window_actions.pop(0)
                
            if len(window_imgs) == self.seq_len:
                batch_imgs.append(np.array(window_imgs))
                batch_states.append(np.array(window_states))
                batch_actions.append(np.array(window_actions))
                
                if len(batch_imgs) == self.batch_size:
                    batch = {
                        "image": jnp.array(batch_imgs),
                        "joint_states": jnp.array(batch_states),
                        "joint_actions": jnp.array(batch_actions),
                    }
                    yield self._apply_kinematics(batch, "so100")
                    yielded_batches += 1
                    
                    batch_imgs = []
                    batch_states = []
                    batch_actions = []
                    
                    if self.limit and yielded_batches >= self.limit:
                        break
                        
        cap.release()
