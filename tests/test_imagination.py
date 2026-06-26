import os
import sys
import numpy as np
import pytest

# Ensure we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import jax
import jax.numpy as jnp
from jepa_robotics.models.world_model import ActionConditionedTransformer
from jepa_robotics.data.kinematics import forward_kinematics

@pytest.fixture
def wm_def():
    """Provides a fresh World Model instance."""
    return ActionConditionedTransformer(
        latent_dim=64, # Small dimensions for fast testing
        depth=2, 
        num_heads=4, 
        activation_fn="gelu"
    )

def test_world_model_shapes(wm_def):
    """
    Test that the ActionConditionedTransformer correctly accepts 
    a history sequence of 5 frames and outputs predictions for those 5 frames.
    """
    batch_size = 2
    seq_len = 5
    latent_dim = 64
    
    # Mock inputs
    mock_seq_context = jnp.ones((batch_size, seq_len, latent_dim))
    mock_seq_actions = jnp.ones((batch_size, seq_len, 10))
    
    init_rng = jax.random.PRNGKey(0)
    
    # Initialize weights
    params = wm_def.init(init_rng, mock_seq_context, mock_seq_actions)
    
    # Forward pass
    next_states = wm_def.apply(params, mock_seq_context, mock_seq_actions)
    
    # Output should exactly match the input sequence length and latent dimension
    assert next_states.shape == (batch_size, seq_len, latent_dim)

def test_causal_masking(wm_def):
    """
    Test that the causal mask prevents future states from influencing past predictions.
    If we change the 5th frame, the prediction for the 1st frame should NOT change!
    """
    seq_len = 5
    latent_dim = 64
    
    mock_seq_context = jax.random.normal(jax.random.PRNGKey(1), (1, seq_len, latent_dim))
    mock_seq_actions = jax.random.normal(jax.random.PRNGKey(2), (1, seq_len, 10))
    
    init_rng = jax.random.PRNGKey(0)
    params = wm_def.init(init_rng, mock_seq_context, mock_seq_actions)
    
    # Forward pass 1
    out_1 = wm_def.apply(params, mock_seq_context, mock_seq_actions)
    
    # Corrupt the 5th frame
    corrupted_context = mock_seq_context.at[:, 4, :].set(999.0)
    corrupted_actions = mock_seq_actions.at[:, 4, :].set(999.0)
    
    # Forward pass 2
    out_2 = wm_def.apply(params, corrupted_context, corrupted_actions)
    
    # The prediction at t=0 should be COMPLETELY UNCHANGED despite t=4 being corrupted.
    # This proves the causal triangular mask is working!
    assert np.allclose(out_1[:, 0, :], out_2[:, 0, :], atol=1e-5)
    
    # But the prediction at t=4 SHOULD change!
    assert not np.allclose(out_1[:, 4, :], out_2[:, 4, :])

def test_kinematics_bridge():
    """
    Ensures the SO100 kinematics bridge outputs the expected 10D tensor 
    that feeds into the World Model.
    """
    joints = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.5]]) # Batch=1
    pose_10d = forward_kinematics(joints, robot_type="so100")
    
    assert pose_10d.shape == (1, 10)
    assert not np.any(np.isnan(pose_10d))
