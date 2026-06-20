import pytest
import jax.numpy as jnp
from jepa_robotics.data.dataset_loaders import BridgeDataLoader, SO100DataLoader

def test_bridge_data_loader_initialization():
    """Test that the BridgeDataLoader initializes correctly with parameters."""
    loader = BridgeDataLoader(limit=5)
    assert loader.limit == 5

def test_bridge_data_loader_yields_correct_structure():
    """Test that the BridgeDataLoader yields the correct JAX array structure."""
    loader = BridgeDataLoader(limit=2)
    iterator = loader.load()
    
    batch = next(iterator)
    
    # Assert all keys are present
    assert "image" in batch
    assert "joint_states" in batch
    assert "joint_actions" in batch
    assert "state_7d" in batch
    assert "action_7d" in batch
    
    # Assert shapes and types
    assert isinstance(batch["image"], jnp.ndarray)
    assert isinstance(batch["state_7d"], jnp.ndarray)
    
    # Cartesian state should be [Batch, 7]
    assert batch["state_7d"].shape == (1, 7)
    
def test_so100_data_loader_initialization():
    """Test that SO100DataLoader initializes correctly."""
    loader = SO100DataLoader(hf_repo="lerobot/svla_so100_stacking", limit=2)
    assert loader.limit == 2

def test_so100_data_loader_yields_correct_structure():
    """Test that SO100DataLoader yields correct dictionary keys and JAX tensors."""
    loader = SO100DataLoader(hf_repo="lerobot/svla_so100_stacking", limit=1)
    iterator = loader.load()
    
    batch = next(iterator)
    
    assert "image" in batch
    assert "joint_states" in batch
    assert "joint_actions" in batch
    assert "state_7d" in batch
    assert "action_7d" in batch
    
    assert batch["state_7d"].shape == (1, 7)
