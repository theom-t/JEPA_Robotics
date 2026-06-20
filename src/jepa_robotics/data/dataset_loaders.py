"""
Object-Oriented dataset loaders for BridgeData V2 and LeRobot SO100.
"""
import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Iterator
from jepa_robotics.data.kinematics import forward_kinematics

class BaseRobotDataset:
    """Base class for managing and loading robot trajectory datasets."""
    def __init__(self, data_dir: str = "./data", limit: Optional[int] = None):
        self.data_dir = data_dir
        self.limit = limit
        
    def load(self) -> Iterator[Dict[str, Any]]:
        """Loads and yields batches of data from the dataset."""
        raise NotImplementedError("Subclasses must implement the load method.")
        
    def _apply_kinematics(self, batch: Dict[str, Any], robot_type: str) -> Dict[str, Any]:
        """
        Passes native joint states through the Kinematic Bridge to standardize output.
        """
        if "joint_states" in batch:
            batch["state_7d"] = forward_kinematics(batch["joint_states"], robot_type)
        if "joint_actions" in batch:
            batch["action_7d"] = forward_kinematics(batch["joint_actions"], robot_type)
        return batch

class BridgeDataLoader(BaseRobotDataset):
    """Loader for BridgeData V2 (WidowX 250)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self) -> Iterator[Dict[str, Any]]:
        import numpy as np
        import cv2
        
        # Load the realistic generated mock image of the WidowX 250
        image_path = "/home/tmainetucker/.gemini/antigravity-cli/brain/aace51ee-6ebc-4b9f-a3fb-73db9385b422/widowx_mock_frame_1781971831939.jpg"
        frame_bgr = cv2.imread(image_path)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (256, 256))
            
        print(f"Loading BridgeData V2 Realistic Stream... (Limit: {self.limit})")
        
        max_items = self.limit if self.limit else 15
        for i in range(max_items):
            # We mock the states here because true BridgeData V2 is a 130GB download
            # that requires a dedicated background Google Cloud sync task.
            mock_batch = {
                "image": jnp.array(np.expand_dims(frame_rgb, axis=0)),
                "joint_states": jnp.array(np.random.normal(size=(1, 7))), # 6 DoF + Gripper
                "joint_actions": jnp.array(np.random.normal(size=(1, 7))),
            }
            yield self._apply_kinematics(mock_batch, "widowx")

class SO100DataLoader(BaseRobotDataset):
    """Loader for LeRobot SO100 Hugging Face datasets."""
    def __init__(self, hf_repo: str = "lerobot/svla_so100_stacking", **kwargs):
        super().__init__(**kwargs)
        self.hf_repo = hf_repo

    def load(self) -> Iterator[Dict[str, Any]]:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
        import cv2
        import numpy as np
        
        split = "train"
        if self.limit is not None:
            split = f"train[:{self.limit}]"
            
        print(f"Loading REAL LeRobot SO100 Data from {self.hf_repo}... (Split: {split})")
        
        # Load the states/actions metadata (Parquet)
        dataset = load_dataset(self.hf_repo, split=split, cache_dir=self.data_dir)
        
        # Fetch the MP4 file manually since the lerobot library requires Python 3.12
        # and we want to maintain our robust Python 3.10 environment
        print("Fetching video chunk from Hugging Face Hub...")
        video_path = hf_hub_download(
            repo_id=self.hf_repo,
            repo_type="dataset",
            filename="videos/observation.images.top/chunk-000/file-000.mp4",
            cache_dir=self.data_dir
        )
        
        cap = cv2.VideoCapture(video_path)
        
        for i, row in enumerate(dataset):
            # Read the corresponding video frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # OpenCV loads as BGR, convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (256, 256))
            
            # Map HF dataset features to our pipeline expectations
            # The SO100 has 6 joints + 1 gripper = 7 DoF
            batch = {
                "image": jnp.array(np.expand_dims(frame_rgb, axis=0)),
                "joint_states": jnp.array(np.expand_dims(row["observation.state"], axis=0)),
                "joint_actions": jnp.array(np.expand_dims(row["action"], axis=0)),
            }
            yield self._apply_kinematics(batch, "so100")
            
        cap.release()
