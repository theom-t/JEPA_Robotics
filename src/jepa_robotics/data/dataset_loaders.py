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
        Standardizes output into 10D Cartesian task space (X, Y, Z, 6D Rot, Gripper).
        """
        from jepa_robotics.data.kinematics import rpy_to_6d, forward_kinematics
        
        if robot_type == "widowx":
            # BridgeData V2 is ALREADY recorded in 7D Cartesian space (X,Y,Z, R,P,Y, Gripper).
            # We bypass DH kinematics and simply convert the raw RPY to 6D rotation!
            def convert_7d_to_10d(states_7d):
                b, s, _ = states_7d.shape
                flat = states_7d.reshape(b * s, 7)
                xyz = flat[:, :3]
                rpy = flat[:, 3:6]
                gripper = flat[:, 6:]
                rot_6d = rpy_to_6d(rpy)
                flat_10d = jnp.concatenate([xyz, rot_6d, gripper], axis=-1)
                return flat_10d.reshape(b, s, 10)
                
            if "joint_states" in batch:
                batch["state_10d"] = convert_7d_to_10d(batch["joint_states"])
            if "joint_actions" in batch:
                batch["action_10d"] = convert_7d_to_10d(batch["joint_actions"])
                
        else:
            # SO100 is recorded in native joint-angles. We must pass it through
            # the Kinematic Bridge to map it to the shared 10D Cartesian space.
            if "joint_states" in batch:
                b, s, dof = batch["joint_states"].shape
                flat_states = batch["joint_states"].reshape(b * s, dof)
                flat_10d = forward_kinematics(flat_states, robot_type)
                batch["state_10d"] = flat_10d.reshape(b, s, 10)
                
            if "joint_actions" in batch:
                b, s, dof = batch["joint_actions"].shape
                flat_actions = batch["joint_actions"].reshape(b * s, dof)
                flat_10d = forward_kinematics(flat_actions, robot_type)
                batch["action_10d"] = flat_10d.reshape(b, s, 10)
                
        return batch

class BridgeDataLoader(BaseRobotDataset):
    """Loader for BridgeData V2 (WidowX 250) using raw TFRecord parsing to bypass tfds."""
    def __init__(self, tfds_data_dir: str = "/home/tmainetucker/Repos/JEPA_Robotics/data/bridge_data_v2", sample_fraction: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.tfds_data_dir = tfds_data_dir
        self.sample_fraction = sample_fraction

    def load(self, split: str = "train") -> Iterator[Dict[str, Any]]:
        import tensorflow as tf
        import glob
        import os
        
        # Only print this the very first time the loader starts
        if not hasattr(self, "_has_logged"):
            print(f"Loading REAL BridgeData V2 (Raw TFRecord) from {self.tfds_data_dir}... (Split: {split}, Fraction: {self.sample_fraction})")
            self._has_logged = True
        
        actual_split = "test" if split == "val" else split
        search_pattern = os.path.join(self.tfds_data_dir, "bridge", "0.1.0", f"bridge-{actual_split}.tfrecord*")
        files = glob.glob(search_pattern)
        
        if not files:
            raise FileNotFoundError(f"No TFRecord files found for split {split} at {search_pattern}")
            
        # Shuffle files deterministically
        files = sorted(files)
        np.random.seed(42)
        np.random.shuffle(files)
        
        # Calculate exactly what the slice of files is
        slice_files = max(1, int(len(files) * self.sample_fraction))
        sliced_files = files[:slice_files]
        
        raw_dataset = tf.data.TFRecordDataset(sliced_files, num_parallel_reads=tf.data.AUTOTUNE)
        
        # Ignore corrupted/partial records so training doesn't crash while gcloud is still downloading
        raw_dataset = raw_dataset.ignore_errors(log_warning=True)
        
        def _parse_function(example_proto):
            feature_description = {
                'steps/observation/image': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
                'steps/observation/state': tf.io.FixedLenSequenceFeature([7], tf.float32, allow_missing=True),
                'steps/action/world_vector': tf.io.FixedLenSequenceFeature([3], tf.float32, allow_missing=True),
                'steps/action/rotation_delta': tf.io.FixedLenSequenceFeature([3], tf.float32, allow_missing=True),
                'steps/action/open_gripper': tf.io.FixedLenSequenceFeature([1], tf.int64, allow_missing=True),
            }
            
            parsed = tf.io.parse_single_example(example_proto, feature_description)
            
            def decode_img(img_str):
                img = tf.io.decode_image(img_str, channels=3, expand_animations=False)
                img = tf.image.resize(img, [256, 256])
                return img / 255.0
                
            images = tf.map_fn(decode_img, parsed['steps/observation/image'], fn_output_signature=tf.float32)
            
            world_vec = parsed['steps/action/world_vector']
            rot_delta = parsed['steps/action/rotation_delta']
            gripper = tf.cast(parsed['steps/action/open_gripper'], tf.float32)
            action = tf.concat([world_vec, rot_delta, gripper], axis=-1)
            
            state = parsed['steps/observation/state']
            
            return images, state, action

        # Map the parsing function
        parsed_dataset = raw_dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Unbatch to flatten the sequence of steps into a flat dataset of steps
        step_ds = parsed_dataset.unbatch()
        
        # Create sliding windows of size seq_len
        window_ds = step_ds.window(self.seq_len, shift=1, drop_remainder=True)
        window_ds = window_ds.flat_map(lambda img, state, action: tf.data.Dataset.zip((img.batch(self.seq_len), state.batch(self.seq_len), action.batch(self.seq_len))))
        
        # Batch the windows
        batched_ds = window_ds.batch(self.batch_size, drop_remainder=True)
        batched_ds = batched_ds.prefetch(tf.data.AUTOTUNE)
        
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
        self._cached_dataset = None

    def _get_dataset(self, split: str):
        import os
        from datasets.utils.logging import set_verbosity_error, disable_progress_bar
        from datasets import load_dataset
        
        if self._cached_dataset is None:
            # Force Hugging Face into strictly offline mode
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            set_verbosity_error()
            disable_progress_bar()
            self._cached_dataset = load_dataset(self.hf_repo, split="train", cache_dir=self.offline_dir)
            
        dataset = self._cached_dataset
        
        # Hard 90/10 split
        total_rows = len(dataset)
        split_idx = int(total_rows * 0.9)
        
        if split == "val":
            dataset = dataset.select(range(split_idx, total_rows))
        else:
            dataset = dataset.select(range(split_idx))
            
        slice_rows = max(1, int(len(dataset) * self.sample_fraction))
        return dataset.shuffle(seed=42).select(range(slice_rows))

    def load(self, split: str = "train") -> Iterator[Dict[str, Any]]:
        import cv2
        
        # Only print this the very first time the loader starts
        if not hasattr(self, "_has_logged"):
            print(f"Loading REAL LeRobot SO100 Data OFFLINE from {self.offline_dir}... (Split: {split}, Fraction: {self.sample_fraction})")
            self._has_logged = True
            
        dataset = self._get_dataset(split)
        
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
            frame_rgb = frame_rgb.astype(np.float32) / 255.0
            
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
