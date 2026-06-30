"""
Object-Oriented dataset loaders for BridgeData V2 and LeRobot SO100.
"""
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
                flat_10d = np.concatenate([xyz, rot_6d, gripper], axis=-1)
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
                img = tf.image.resize(img, [512, 512])
                return tf.cast(img, tf.uint8)
                
            images = tf.map_fn(decode_img, parsed['steps/observation/image'], fn_output_signature=tf.uint8)
            
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
        # Cap tf.data prefetch to 2 batches. AUTOTUNE will eagerly devour all 32GB of RAM 
        # since each 128-size float32 batch is ~603 MB!
        batched_ds = batched_ds.prefetch(2)
        
        if self.limit:
            batched_ds = batched_ds.take(self.limit)
            
        for imgs, states, actions in batched_ds.as_numpy_iterator():
            batch = {
                "image": imgs,                     # (B, S, 512, 512, 3) already contiguous!
                "joint_states": states,            # (B, S, 7) for WidowX
                "joint_actions": actions           # (B, S, 7) for WidowX
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
        
        # Deadlock prevention: Ensure the slice is large enough to form at least ONE batch 
        # (useful for --fast test runs where 5% of the val split might be < 128 frames)
        min_required_frames = self.batch_size + self.seq_len
        slice_rows = min(len(dataset), max(slice_rows, min_required_frames))
        
        return dataset.shuffle(seed=42).select(range(slice_rows))

    def load(self, split: str = "train") -> Iterator[Dict[str, Any]]:
        import cv2
        
        # Only print this the very first time the loader starts
        if not hasattr(self, "_has_logged"):
            print(f"Loading REAL LeRobot SO100 Data OFFLINE from {self.offline_dir}... (Split: {split}, Fraction: {self.sample_fraction})")
            self._has_logged = True
            
        dataset = self._get_dataset(split)
        
        # For HF LeRobot datasets, the video is chunked. 
        # We spawn a dedicated C++ decoding thread to prevent cv2 from blocking Python.
        video_path = f"{self.offline_dir}/videos/observation.images.top/chunk-000/file-000.mp4"
        cap = cv2.VideoCapture(video_path)
        
        import queue
        import threading
        # Increase queue size to give cv2 a huge decoding buffer
        frame_queue = queue.Queue(maxsize=1024)
        shutdown_event = threading.Event()
        
        def video_reader():
            try:
                while not shutdown_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.resize(frame, (512, 512))
                    
                    # CRITICAL PERFORMANCE FIX: cv2.cvtColor is incredibly slow because it allocates
                    # a new memory array for every frame. By using a numpy slice, we perform a zero-copy
                    # BGR-to-RGB conversion instantly. This triples the video decoding speed!
                    frame_rgb = frame[:, :, ::-1]
                    # Leave as uint8 to prevent massive CPU division bottleneck!
                    # Conversion to float32 happens on the GPU via JAX.
                    
                    # Use timeout to periodically check shutdown_event if queue is full
                    while not shutdown_event.is_set():
                        try:
                            frame_queue.put(frame_rgb, timeout=0.1)
                            break
                        except queue.Full:
                            continue
            finally:
                cap.release()
                
        reader_thread = threading.Thread(target=video_reader, daemon=True)
        reader_thread.start()
        
        # Convert the HF dataset to numpy format to eliminate PyArrow dictionary deserialization overhead
        dataset = dataset.with_format("numpy")
        
        # Explicitly cast to numpy array in case the dataset object returns a PyArrow Column
        all_states = np.array(dataset["observation.state"])
        all_actions = np.array(dataset["action"])
        total_rows = len(dataset)
        
        try:
            dof = all_states.shape[-1]
            
            # Pre-allocate zero-copy numpy arrays to prevent Python memory copies and GIL locks
            batch_imgs = np.empty((self.batch_size, self.seq_len, 512, 512, 3), dtype=np.uint8)
            batch_states = np.empty((self.batch_size, self.seq_len, dof), dtype=np.float32)
            batch_actions = np.empty((self.batch_size, self.seq_len, dof), dtype=np.float32)
            
            window_imgs = []
            batch_idx = 0
            yielded_batches = 0
            
            # Pre-fill initial window
            for i in range(self.seq_len - 1):
                if i >= total_rows: break
                try:
                    frame = frame_queue.get(timeout=5.0)
                    window_imgs.append(frame)
                except queue.Empty:
                    break
                    
            for i in range(self.seq_len - 1, total_rows):
                try:
                    frame = frame_queue.get(timeout=5.0)
                except queue.Empty:
                    break
                
                window_imgs.append(frame)
                if len(window_imgs) > self.seq_len:
                    window_imgs.pop(0)
                    
                # Insert directly into the preallocated C-arrays
                for step_idx in range(self.seq_len):
                    batch_imgs[batch_idx, step_idx] = window_imgs[step_idx]
                
                start_idx = i - self.seq_len + 1
                end_idx = i + 1
                batch_states[batch_idx] = all_states[start_idx:end_idx]
                batch_actions[batch_idx] = all_actions[start_idx:end_idx]
                
                batch_idx += 1
                
                if batch_idx == self.batch_size:
                    batch = {
                        "image": batch_imgs,
                        "joint_states": batch_states,
                        "joint_actions": batch_actions,
                    }
                    yield self._apply_kinematics(batch, "so100")
                    yielded_batches += 1
                    
                    # Allocate fresh array for the next batch (very fast single memcpy in C)
                    batch_idx = 0
                    batch_imgs = np.empty((self.batch_size, self.seq_len, 512, 512, 3), dtype=np.uint8)
                    batch_states = np.empty((self.batch_size, self.seq_len, dof), dtype=np.float32)
                    batch_actions = np.empty((self.batch_size, self.seq_len, dof), dtype=np.float32)
                    
                    if self.limit and yielded_batches >= self.limit:
                        break
        finally:
            # Guarantee graceful thread shutdown and memory cleanup
            shutdown_event.set()
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    break
            reader_thread.join(timeout=1.0)
