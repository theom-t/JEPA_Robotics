import pytest
import jax
import jax.numpy as jnp
from jepa_robotics.data.kinematics import forward_kinematics

def test_forward_kinematics_shape():
    """
    Test that forward_kinematics correctly transforms an arbitrary 
    joint angle array into a strict (Batch, 7) Cartesian array.
    """
    batch_size = 4
    # Mocking a 6-DoF arm with 1 gripper state (7 total joints)
    num_joints = 7
    
    mock_joints = jnp.zeros((batch_size, num_joints))
    
    cartesian_out = forward_kinematics(mock_joints, "widowx")
    
    assert cartesian_out.shape == (batch_size, 7), f"Expected shape {(batch_size, 7)} but got {cartesian_out.shape}"

def test_forward_kinematics_preserves_gripper():
    """
    Test that the gripper state (the last joint value) is perfectly 
    preserved and mapped to the 7th dimension of the Cartesian vector.
    """
    batch_size = 2
    num_joints = 7
    
    # Set the gripper state (last element) to specific values
    mock_joints = jnp.array([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0], # Gripper open
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]  # Gripper closed
    ])
    
    cartesian_out = forward_kinematics(mock_joints, "so100")
    
    # Gripper state is the 7th element (index 6)
    assert jnp.allclose(cartesian_out[0, 6], 1.0)
    assert jnp.allclose(cartesian_out[1, 6], 0.0)
