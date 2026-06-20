"""
Kinematic processing module for mapping joint-space actions to 7D Cartesian task space.
"""
import jax
import jax.numpy as jnp
import numpy as np

def dh_transform(theta: jax.Array, d: float, a: float, alpha: float) -> jax.Array:
    """Computes batch 4x4 DH transformation matrices."""
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    cos_a = jnp.cos(alpha)
    sin_a = jnp.sin(alpha)
    
    batch_size = theta.shape[0]
    zeros = jnp.zeros(batch_size)
    ones = jnp.ones(batch_size)
    
    r00 = cos_t
    r01 = -sin_t * cos_a
    r02 = sin_t * sin_a
    r03 = a * cos_t
    
    r10 = sin_t
    r11 = cos_t * cos_a
    r12 = -cos_t * sin_a
    r13 = a * sin_t
    
    r20 = zeros
    r21 = jnp.full_like(zeros, sin_a)
    r22 = jnp.full_like(zeros, cos_a)
    r23 = jnp.full_like(zeros, d)
    
    r30 = zeros
    r31 = zeros
    r32 = zeros
    r33 = ones
    
    row0 = jnp.stack([r00, r01, r02, r03], axis=-1)
    row1 = jnp.stack([r10, r11, r12, r13], axis=-1)
    row2 = jnp.stack([r20, r21, r22, r23], axis=-1)
    row3 = jnp.stack([r30, r31, r32, r33], axis=-1)
    
    return jnp.stack([row0, row1, row2, row3], axis=1)

def compute_kinematics_chain(joint_angles: jax.Array, dh_params: list) -> jax.Array:
    batch_size = joint_angles.shape[0]
    T_final = jnp.tile(jnp.eye(4), (batch_size, 1, 1))
    
    for i, (d, a, alpha, offset) in enumerate(dh_params):
        if i < joint_angles.shape[1]:
            theta = joint_angles[:, i] + offset
        else:
            theta = jnp.full((batch_size,), offset)
            
        T_i = dh_transform(theta, d, a, alpha)
        T_final = jnp.matmul(T_final, T_i)
        
    return T_final

def extract_xyz_rpy(T: jax.Array) -> jax.Array:
    """Extracts (X, Y, Z, Roll, Pitch, Yaw) from batch of 4x4 matrices."""
    x = T[:, 0, 3]
    y = T[:, 1, 3]
    z = T[:, 2, 3]
    
    pitch = jnp.arcsin(jnp.clip(-T[:, 2, 0], -1.0, 1.0))
    roll = jnp.arctan2(T[:, 2, 1], T[:, 2, 2])
    yaw = jnp.arctan2(T[:, 1, 0], T[:, 0, 0])
    
    return jnp.stack([x, y, z, roll, pitch, yaw], axis=-1)

def forward_kinematics(joint_angles: jax.Array, robot_type: str) -> jax.Array:
    """
    Transforms joint angles into 7D Cartesian space using DH parameters.
    Returns: (Batch, 7) [X, Y, Z, roll, pitch, yaw, gripper_state]
    """
    gripper_state = joint_angles[:, -1:]
    
    if robot_type == "widowx":
        beta = 0.2007 # 11.5 deg dog-leg offset
        dh_table = [
            (0.11, 0.0, -np.pi/2, 0.0),
            (0.0, -0.25, 0.0, np.pi/2 + beta),
            (0.0, 0.0, np.pi/2, -beta),
            (0.50, 0.0, -np.pi/2, 0.0),
            (0.0, 0.0, np.pi/2, np.pi),
            (0.11, 0.0, 0.0, 0.0),
        ]
    else: # so100
        dh_table = [
            (0.08, 0.0, np.pi/2, 0.0),
            (0.0, 0.10, 0.0, 0.0),
            (0.0, 0.10, 0.0, 0.0),
            (0.0, 0.05, -np.pi/2, 0.0),
            (0.05, 0.0, 0.0, 0.0),
        ]
        
    T_final = compute_kinematics_chain(joint_angles, dh_table)
    xyz_rpy = extract_xyz_rpy(T_final)
    
    if robot_type == "so100":
        # SO100 gripper is a raw motor degree, squash it so it matches BridgeData [0, 1] range
        gripper_state = jnp.tanh(gripper_state / 100.0)
        
    cartesian_7d = jnp.concatenate([xyz_rpy, gripper_state], axis=1)
    
    return cartesian_7d
