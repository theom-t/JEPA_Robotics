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
    """Loader for the BridgeData V2 RLDS dataset."""
    def load(self) -> Iterator[Dict[str, Any]]:
        import tensorflow_datasets as tfds
        import tensorflow as tf
        
        print(f"Loading BridgeData V2... (Limit: {self.limit})")
        # In practice, we'd load the builder or specific tfds path here.
        # dataset = tfds.load('bridge_dataset', data_dir=self.data_dir, split='train')
        
        # Mocking the generator for pipeline testing
        max_items = self.limit if self.limit else 5
        for i in range(max_items):
            mock_batch = {
                "image": jnp.zeros((1, 256, 256, 3), dtype=jnp.uint8),
                "joint_states": jnp.zeros((1, 8)), # 7 DoF + Gripper
                "joint_actions": jnp.zeros((1, 8)),
            }
            yield self._apply_kinematics(mock_batch, "widowx")

class SO100DataLoader(BaseRobotDataset):
    """Loader for LeRobot SO100 Hugging Face datasets."""
    def __init__(self, hf_repo: str = "lerobot/svla_so100_stacking", **kwargs):
        super().__init__(**kwargs)
        self.hf_repo = hf_repo

    def load(self) -> Iterator[Dict[str, Any]]:
        # Import inside the method to keep initialization light
        from datasets import load_dataset
        
        split = "train"
        if self.limit is not None:
            split = f"train[:{self.limit}]"
            
        print(f"Loading LeRobot SO100 Data from {self.hf_repo}... (Split: {split})")
        # dataset = load_dataset(self.hf_repo, split=split, cache_dir=self.data_dir)
        # dataset.set_format("jax")
        
        # Mocking the generator for pipeline testing
        max_items = self.limit if self.limit else 5
        for i in range(max_items):
            mock_batch = {
                "image": jnp.zeros((1, 256, 256, 3), dtype=jnp.uint8),
                "joint_states": jnp.zeros((1, 7)), # 6 DoF + Gripper
                "joint_actions": jnp.zeros((1, 7)),
            }
            yield self._apply_kinematics(mock_batch, "so100")
