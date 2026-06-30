import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.stress_test import apply_targeted_occlusion

def test_apply_targeted_occlusion():
    # Create a fake 256x256 image with all ones
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255
    
    # Apply occlusion
    occluded = apply_targeted_occlusion(img)
    
    # Verify shape remains the same
    assert occluded.shape == img.shape
    
    # Verify there is a black box (some pixels are exactly 0)
    zero_pixels = np.sum(occluded == 0)
    # 64x64x3 = 12288
    assert zero_pixels == 12288

def test_jerk_math():
    # A perfectly smooth constant velocity should have 0 jerk
    # Positions = [0, 1, 2, 3, 4]
    trajectory = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0]
    ])
    
    velocities = np.diff(trajectory, axis=0) # [1, 1, 1, 1]
    accelerations = np.diff(velocities, axis=0) # [0, 0, 0]
    jerks = np.diff(accelerations, axis=0) # [0, 0]
    
    mean_jerk = float(np.mean(np.linalg.norm(jerks, axis=1)))
    assert mean_jerk == 0.0

def test_teleportation_math():
    MAX_VELOCITY = 0.2
    
    # Valid move
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.1, 0.0, 0.0])
    assert np.linalg.norm(p2 - p1) < MAX_VELOCITY
    
    # Teleportation
    p3 = np.array([0.5, 0.0, 0.0])
    assert np.linalg.norm(p3 - p1) > MAX_VELOCITY

def test_collision_math():
    TABLE_Z_HEIGHT = 0.0
    
    # Safe
    p_safe = np.array([0.5, 0.5, 0.1])
    assert p_safe[2] >= TABLE_Z_HEIGHT
    
    # Collision
    p_crash = np.array([0.5, 0.5, -0.1])
    assert p_crash[2] < TABLE_Z_HEIGHT
