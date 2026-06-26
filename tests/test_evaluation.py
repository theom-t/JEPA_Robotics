import os
import sys
import numpy as np
import pytest

# Suppress MuJoCo headless rendering warnings
os.environ["MUJOCO_GL"] = "egl"

# Ensure we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from jepa_robotics.evaluation.mujoco_env import SO100SimEnv, SO100GhostEnv
from jepa_robotics.evaluation.ik_solver import IKSolver
from jepa_robotics.data.kinematics import forward_kinematics

@pytest.fixture
def sim_env():
    """Provides a fresh SO100SimEnv instance."""
    return SO100SimEnv()

@pytest.fixture
def ghost_env():
    """Provides a fresh SO100GhostEnv instance."""
    return SO100GhostEnv()

@pytest.fixture
def ik_solver():
    """Provides a fresh IKSolver instance."""
    return IKSolver()


def test_sim_env_initialization(sim_env):
    """Test that the true physics environment initializes correctly."""
    assert sim_env.model is not None
    assert sim_env.data is not None
    assert sim_env.renderer is not None

def test_sim_env_reset_and_kinematics(sim_env):
    """Test resetting the arm and getting the 10D ground truth pose."""
    test_joints = np.array([0.5, -0.5, 0.2, 0.0, 1.0, 0.5])
    sim_env.reset(test_joints)
    
    # Check joints updated in physics engine
    for i in range(6):
        assert np.isclose(sim_env.data.qpos[i], test_joints[i], atol=1e-4)
        
    # Get 10D Ground truth
    gt_10d = sim_env.get_ground_truth_10d()
    
    assert gt_10d.shape == (10,)
    assert not np.any(np.isnan(gt_10d))
    # Gripper bounds squashed by tanh
    assert -1.0 <= gt_10d[9] <= 1.0

def test_sim_env_render(sim_env):
    """Test that the environment renders a normalized 256x256x3 float32 image."""
    sim_env.reset(np.zeros(6))
    img = sim_env.render()
    
    assert img.shape == (256, 256, 3)
    assert img.dtype == np.float32
    assert np.max(img) <= 1.0
    assert np.min(img) >= 0.0

def test_ghost_env_overlay(sim_env, ghost_env):
    """Test that the holographic ghost renderer correctly blends images."""
    sim_env.reset(np.zeros(6))
    base_img = sim_env.render()
    
    # Render ghost in a completely different pose
    ghost_joints = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    blended_img = ghost_env.overlay_ghost(base_img, ghost_joints)
    
    assert blended_img.shape == (256, 256, 3)
    assert blended_img.dtype == np.float32
    assert np.max(blended_img) <= 1.0
    assert np.min(blended_img) >= 0.0
    # Ensure it's not identical to the base image (blending occurred)
    assert not np.array_equal(base_img, blended_img)

def test_ik_solver_accuracy(ik_solver):
    """
    Test that the Inverse Kinematics solver can accurately reverse-engineer 
    a 10D Cartesian pose back into valid joint angles.
    """
    # 1. Define a random valid joint target
    target_joints = np.array([0.3, -0.4, 0.5, 0.1, -0.2, 0.5])
    
    # 2. Forward Kinematics to get the true 10D target
    joints_batch = np.expand_dims(target_joints, axis=0)
    target_10d = forward_kinematics(joints_batch, robot_type="so100")[0]
    
    # 3. Solve IK
    # We provide a nearby initial guess to speed up convergence
    initial_guess = target_joints + np.random.uniform(-0.2, 0.2, size=(6,))
    solved_joints = ik_solver.solve(target_10d, initial_guess=initial_guess)
    
    assert solved_joints.shape == (6,)
    assert not np.any(np.isnan(solved_joints))
    
    # 4. Verify the solved joints produce the SAME 10D pose mathematically!
    solved_batch = np.expand_dims(solved_joints, axis=0)
    solved_10d = forward_kinematics(solved_batch, robot_type="so100")[0]
    
    # Position should be extremely accurate (MSE < 1e-4)
    pos_mse = np.mean((solved_10d[:3] - target_10d[:3])**2)
    assert pos_mse < 1e-4
    
    # Rotation should be highly accurate
    rot_mse = np.mean((solved_10d[3:9] - target_10d[3:9])**2)
    assert rot_mse < 1e-3
