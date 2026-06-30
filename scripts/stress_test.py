import os
import sys
# Suppress XLA C++ warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_PLATFORMS"] = "cuda,cpu"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import jax
import jax.numpy as jnp
import numpy as np
import flax.serialization
from tqdm import tqdm
import json
import cv2

from jepa_robotics.evaluation.mujoco_env import SO100SimEnv
from jepa_robotics.models.v_jepa import ViTEncoder, StateLinearProbe
from jepa_robotics.models.world_model import ActionConditionedTransformer
from jepa_robotics.data.kinematics import forward_kinematics

# V1 Config Baseline
CONFIG = {
    "latent_dim": 256,
    "vit_depth": 4,
    "patch_size": 16,
    "wm_depth": 4,
    "num_heads": 16,
    "activation_fn": "gelu"
}

def load_models(weights_path):
    print(f"[INFO] Loading V-JEPA Models from {weights_path}...")
    
    encoder_def = ViTEncoder(
        latent_dim=CONFIG["latent_dim"], 
        depth=CONFIG["vit_depth"], 
        num_heads=CONFIG["num_heads"],
        patch_size=CONFIG["patch_size"], 
        activation_fn=CONFIG["activation_fn"]
    )
    probe_def = StateLinearProbe(out_dim=10)
    wm_def = ActionConditionedTransformer(
        latent_dim=CONFIG["latent_dim"], 
        depth=CONFIG["wm_depth"], 
        num_heads=CONFIG["num_heads"], 
        activation_fn=CONFIG["activation_fn"]
    )
    
    with open(weights_path, "rb") as f:
        loaded_state = flax.serialization.msgpack_restore(f.read())
        
    return encoder_def, probe_def, wm_def, loaded_state

def apply_targeted_occlusion(img):
    """Draws a 64x64 black box randomly near the center to simulate heavy object/gripper occlusion."""
    occluded = img.copy()
    h, w = occluded.shape[:2]
    # Keep it near the center where the arm usually is
    cx = np.random.randint(w//4, 3*w//4)
    cy = np.random.randint(h//4, 3*h//4)
    size = 64
    x1 = max(0, cx - size//2)
    x2 = min(w, cx + size//2)
    y1 = max(0, cy - size//2)
    y2 = min(h, cy + size//2)
    occluded[y1:y2, x1:x2, :] = 0
    return occluded

def run_occlusion_stress_test(env, encoder_def, probe_def, loaded_state, num_trials=100):
    print(f"\n--- PHASE 1: Targeted Occlusion Permanence (Trials: {num_trials}) ---")
    
    @jax.jit
    def perceive(img):
        _, pooled = encoder_def.apply({'params': loaded_state['encoder_params']['params']}, img, patch_indices=None)
        pred_10d = probe_def.apply({'params': loaded_state['probe_params']['params']}, pooled)
        return pred_10d
        
    baseline_mses = []
    occluded_mses = []
    
    for i in tqdm(range(num_trials), desc="Testing Occlusion Robustness"):
        random_joints = np.random.uniform(-1.0, 1.0, size=(6,))
        env.reset(random_joints)
        gt_10d = env.get_ground_truth_10d()
        
        env.cam.azimuth = 180
        env.cam.lookat[:] = [0.25, 0.0, 0.0]
        golden_img = env.render()
        
        # Baseline Predict
        golden_img_batch = jnp.expand_dims(golden_img, axis=0)
        pred_golden = perceive(golden_img_batch)[0]
        baseline_mse = np.mean((np.array(pred_golden) - gt_10d) ** 2)
        baseline_mses.append(baseline_mse)
        
        # Occluded Predict
        occluded_img = apply_targeted_occlusion(golden_img)
        occluded_img_batch = jnp.expand_dims(occluded_img, axis=0)
        pred_occluded = perceive(occluded_img_batch)[0]
        occluded_mse = np.mean((np.array(pred_occluded) - gt_10d) ** 2)
        occluded_mses.append(occluded_mse)
        
    avg_base = float(np.mean(baseline_mses))
    avg_occ = float(np.mean(occluded_mses))
    deg_factor = avg_occ / (avg_base + 1e-8)
    
    print(f"\nResults:")
    print(f"  Average Baseline MSE:  {avg_base:.6f}")
    print(f"  Average Occluded MSE:  {avg_occ:.6f}")
    print(f"  Occlusion Degradation Factor: {deg_factor:.2f}x")
    
    return {
        "baseline_mse": avg_base,
        "occluded_mse": avg_occ,
        "degradation_factor": deg_factor
    }

def run_crucible_imagination_test(env, encoder_def, wm_def, probe_def, loaded_state, num_trials=10):
    print(f"\n--- PHASE 2: Crucible Physics Audit (Trials: {num_trials}) ---")
    
    @jax.jit
    def encode(img):
        _, pooled = encoder_def.apply({'params': loaded_state['encoder_params']['params']}, img, patch_indices=None)
        return pooled
        
    @jax.jit
    def imagine_step(seq_context, seq_actions):
        next_states = wm_def.apply({'params': loaded_state['wm_params']['params']}, seq_context, seq_actions)
        return next_states[:, -1:, :]
        
    @jax.jit
    def decode(pooled):
        return probe_def.apply({'params': loaded_state['probe_params']['params']}, pooled)

    total_teleportations = 0
    total_collisions = 0
    jerk_metrics = []
    
    MAX_VELOCITY_PER_STEP = 0.2  # 20cm max move per 33ms step
    TABLE_Z_HEIGHT = 0.0         # Anything below 0 is a collision

    for i in tqdm(range(num_trials), desc="Physics Crucible"):
        start_joints = np.random.uniform(-1.0, 1.0, size=(6,))
        env.reset(start_joints)
        
        img = env.render()
        img_batch = jnp.expand_dims(img, axis=0)
        latent_state = encode(img_batch)
        
        context_buffer = jnp.repeat(jnp.expand_dims(latent_state, 1), 5, axis=1)
        start_10d = forward_kinematics(np.expand_dims(start_joints, axis=0), robot_type="so100")[0]
        action_buffer = jnp.repeat(jnp.expand_dims(jnp.expand_dims(jnp.array(start_10d), 0), 1), 5, axis=1)
        
        action_vel = np.random.uniform(-0.1, 0.1, size=(6,))
        
        trajectory_positions = [start_10d[:3]] # X, Y, Z
        
        for step in range(10):
            action_joints = start_joints + action_vel * (step + 1)
            action_joints = np.clip(action_joints, -1.5, 1.5)
            action_10d = forward_kinematics(np.expand_dims(action_joints, axis=0), robot_type="so100")[0]
            
            action_expanded = jnp.expand_dims(jnp.expand_dims(jnp.array(action_10d), 0), 1)
            action_buffer = jnp.concatenate([action_buffer[:, 1:, :], action_expanded], axis=1)
            
            next_latent = imagine_step(context_buffer, action_buffer)
            pred_10d = np.array(decode(next_latent[0])[0])
            
            pred_xyz = pred_10d[:3]
            prev_xyz = trajectory_positions[-1]
            
            # 1. Teleportation Check (Velocity limit)
            dist = np.linalg.norm(pred_xyz - prev_xyz)
            if dist > MAX_VELOCITY_PER_STEP:
                total_teleportations += 1
                
            # 2. Collision Check (Z < 0)
            if pred_xyz[2] < TABLE_Z_HEIGHT:
                total_collisions += 1
                
            trajectory_positions.append(pred_xyz)
            context_buffer = jnp.concatenate([context_buffer[:, 1:, :], next_latent], axis=1)
            
        # 3. Jerk Analysis (3rd derivative of position)
        traj = np.array(trajectory_positions) # (11, 3)
        velocities = np.diff(traj, axis=0)
        accelerations = np.diff(velocities, axis=0)
        jerks = np.diff(accelerations, axis=0)
        mean_jerk = float(np.mean(np.linalg.norm(jerks, axis=1)))
        jerk_metrics.append(mean_jerk)

    avg_jerk = float(np.mean(jerk_metrics))
    
    print("\nPhysics Audit Results:")
    print(f"  Total Teleportation Violations: {total_teleportations}")
    print(f"  Total Table Collision Violations: {total_collisions}")
    print(f"  Average Kinematic Jerk: {avg_jerk:.6f}")
    
    return {
        "teleportation_violations": total_teleportations,
        "collision_violations": total_collisions,
        "average_jerk": avg_jerk
    }

if __name__ == "__main__":
    weights_path = "/home/tmainetucker/Repos/JEPA_Robotics/models/v1_jepa_backbone/v1_final_weights.msgpack"
    if not os.path.exists(weights_path):
        # fallback
        weights_path = "/home/tmainetucker/Repos/JEPA_Robotics/checkpoints/v1_jepa_backbone/checkpoint_epoch_100.msgpack"
    
    if not os.path.exists(weights_path):
        print(f"Weights not found. Please ensure V1 is trained.")
        sys.exit(1)
        
    env = SO100SimEnv()
    encoder_def, probe_def, wm_def, loaded_state = load_models(weights_path)
    
    metrics = {}
    metrics["perception"] = run_occlusion_stress_test(env, encoder_def, probe_def, loaded_state, num_trials=100)
    metrics["imagination"] = run_crucible_imagination_test(env, encoder_def, wm_def, probe_def, loaded_state, num_trials=10)
    
    with open("crucible_baseline.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("\n[INFO] Saved crucible_baseline.json")
