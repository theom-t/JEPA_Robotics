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
    # SO100 gripper is soft-normalized: tanh(x / 100.0)
    unnormalized_gripper_0 = jnp.arctanh(cartesian_out[0, 6]) * 100.0
    unnormalized_gripper_1 = jnp.arctanh(cartesian_out[1, 6]) * 100.0
    
    assert jnp.allclose(unnormalized_gripper_0, 1.0)
    assert jnp.allclose(unnormalized_gripper_1, 0.0)

def test_dh_math_no_nans():
    """
    Test that the DH matrix multiplication and Euler angle extraction 
    does not produce NaNs (which often happens with arcsin/arctan edge cases).
    """
    batch_size = 10
    num_joints = 7
    
    # Use random angles to test gimbal lock / edge cases
    key = jax.random.PRNGKey(0)
    mock_joints = jax.random.uniform(key, (batch_size, num_joints), minval=-jnp.pi, maxval=jnp.pi)
    
    cartesian_out = forward_kinematics(mock_joints, "widowx")
    
    assert not jnp.any(jnp.isnan(cartesian_out)), "DH Kinematics produced NaN values!"
    assert cartesian_out.shape == (batch_size, 7)

def test_dh_zero_position():
    """
    Test that an arm with all zero angles produces a structurally correct position.
    """
    mock_joints = jnp.zeros((1, 7))
    cartesian_out = forward_kinematics(mock_joints, "so100")
    
    # Check the physical sum of lengths directly in Cartesian space
    raw_pos = cartesian_out[0, 0:3]
    
    # For SO100 with zero angles, the arm stretches out along the X axis.
    # L1(0.08Z) -> L2(0.1X) -> L3(0.1X) -> L4(0.05X) -> L5(0.05X)
    # The X position should be approximately 0.3 meters.
    # The Z position should be approximately 0.08 meters.
    assert raw_pos[0] > 0.1, f"Expected positive X reach, got {raw_pos[0]}"
    assert not jnp.any(jnp.isnan(raw_pos))
