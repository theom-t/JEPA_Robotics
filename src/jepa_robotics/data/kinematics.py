"""
Kinematic processing module for mapping joint-space actions to 7D Cartesian task space.
"""
import jax
import numpy as np
import numpy as jnp  # Alias to reuse the same code but force execution on CPU

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

def extract_xyz_6d(T: jax.Array) -> jax.Array:
    """Extracts (X, Y, Z, 6D Rotation) from batch of 4x4 matrices."""
    x = T[:, 0, 3]
    y = T[:, 1, 3]
    z = T[:, 2, 3]
    
    # 6D continuous rotation uses the first two columns of the rotation matrix
    col1_x = T[:, 0, 0]
    col1_y = T[:, 1, 0]
    col1_z = T[:, 2, 0]
    
    col2_x = T[:, 0, 1]
    col2_y = T[:, 1, 1]
    col2_z = T[:, 2, 1]
    
    return jnp.stack([x, y, z, col1_x, col1_y, col1_z, col2_x, col2_y, col2_z], axis=-1)

def rpy_to_6d(rpy: jax.Array) -> jax.Array:
    """Converts (Roll, Pitch, Yaw) Euler angles to 6D continuous rotation."""
    r = rpy[..., 0]
    p = rpy[..., 1]
    y = rpy[..., 2]
    
    cos_r = jnp.cos(r)
    sin_r = jnp.sin(r)
    cos_p = jnp.cos(p)
    sin_p = jnp.sin(p)
    cos_y = jnp.cos(y)
    sin_y = jnp.sin(y)
    
    r00 = cos_p * cos_y
    r10 = cos_p * sin_y
    r20 = -sin_p
    
    r01 = sin_r * sin_p * cos_y - cos_r * sin_y
    r11 = sin_r * sin_p * sin_y + cos_r * cos_y
    r21 = sin_r * cos_p
    
    col1 = jnp.stack([r00, r10, r20], axis=-1)
    col2 = jnp.stack([r01, r11, r21], axis=-1)
    
    return jnp.concatenate([col1, col2], axis=-1)

def forward_kinematics(joint_angles: jax.Array, robot_type: str) -> jax.Array:
    """
    Transforms joint angles into 10D Cartesian space using DH parameters.
    Returns: (Batch, 10) [X, Y, Z, r1, r2, r3, r4, r5, r6, gripper_state]
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
    xyz_6d = extract_xyz_6d(T_final)
    
    if robot_type == "so100":
        # SO100 gripper is a raw motor degree, squash it so it matches BridgeData [0, 1] range
        gripper_state = jnp.tanh(gripper_state / 100.0)
        
    cartesian_10d = jnp.concatenate([xyz_6d, gripper_state], axis=1)
    
    return cartesian_10d
