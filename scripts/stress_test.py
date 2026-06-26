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

from jepa_robotics.evaluation.mujoco_env import SO100SimEnv
from jepa_robotics.models.v_jepa import ViTEncoder, StateLinearProbe
from jepa_robotics.models.world_model import ActionConditionedTransformer

# V1 Config Baseline
CONFIG = {
    "latent_dim": 256,
    "vit_depth": 7,
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

def run_perception_stress_test(env, encoder_def, probe_def, loaded_state, num_trials=1000):
    print(f"\n--- PHASE 1: Perception Spatial Robustness (Trials: {num_trials}) ---")
    
    # JIT Compile the perception step
    @jax.jit
    def perceive(img):
        # We pass the full image through the context encoder (no masking) 
        # to match how the probe was gradient-attached during training.
        _, pooled = encoder_def.apply({'params': loaded_state['encoder_params']['params']}, img, patch_indices=None)
        pred_10d = probe_def.apply({'params': loaded_state['probe_params']}, pooled)
        return pred_10d
        
    baseline_mses = []
    perturbed_mses = []
    
    for i in tqdm(range(num_trials), desc="Stress Testing Perception"):
        # 1. Place robot in a random valid configuration
        # SO100 joint limits are approximately -1.5 to 1.5 radians
        random_joints = np.random.uniform(-1.0, 1.0, size=(6,))
        env.reset(random_joints)
        
        # 2. Get true 10D pose from physical simulation
        gt_10d = env.get_ground_truth_10d()
        
        # 3. Render In-Distribution Baseline (Camera 1B)
        env.cam.azimuth = 180
        env.cam.lookat[:] = [0.25, 0.0, 0.0]
        golden_img = env.render()
        
        # Predict
        golden_img_batch = jnp.expand_dims(golden_img, axis=0) # (1, 256, 256, 3)
        pred_golden = perceive(golden_img_batch)[0]
        
        baseline_mse = np.mean((np.array(pred_golden) - gt_10d) ** 2)
        baseline_mses.append(baseline_mse)
        
        # 4. Render Out-of-Distribution Perturbation
        # We add uniform noise to shift the camera geometrically 
        env.cam.azimuth = 180 + np.random.uniform(-15.0, 15.0) # +/- 15 deg azimuth noise
        env.cam.lookat[0] += np.random.uniform(-0.05, 0.05)    # +/- 5cm X noise
        env.cam.lookat[1] += np.random.uniform(-0.05, 0.05)    # +/- 5cm Y noise
        
        perturbed_img = env.render()
        perturbed_img_batch = jnp.expand_dims(perturbed_img, axis=0)
        pred_perturbed = perceive(perturbed_img_batch)[0]
        
        perturbed_mse = np.mean((np.array(pred_perturbed) - gt_10d) ** 2)
        perturbed_mses.append(perturbed_mse)
        
    avg_base = float(np.mean(baseline_mses))
    avg_pert = float(np.mean(perturbed_mses))
    deg_factor = avg_pert / (avg_base + 1e-8)
    
    print(f"\nResults:")
    print(f"  Average Baseline MSE (In-Distribution):  {avg_base:.6f}")
    print(f"  Average Perturbed MSE (Out-Distribution): {avg_pert:.6f}")
    print(f"  Degradation Factor: {deg_factor:.2f}x")
    
    return {
        "baseline_mse": avg_base,
        "perturbed_mse": avg_pert,
        "degradation_factor": deg_factor
    }

def run_imagination_test(env, encoder_def, wm_def, probe_def, loaded_state, num_trials=100):
    print(f"\n--- PHASE 2: World Model Temporal Forecasting (Trials: {num_trials}) ---")
    
    @jax.jit
    def encode(img):
        _, pooled = encoder_def.apply({'params': loaded_state['encoder_params']['params']}, img, patch_indices=None)
        return pooled
        
    @jax.jit
    def imagine_step(seq_context, seq_actions):
        # The World Model expects exactly 5 frames of history due to its positional embeddings
        # seq_context: (B, 5, D), seq_actions: (B, 5, 10)
        next_states = wm_def.apply({'params': loaded_state['wm_params']}, seq_context, seq_actions)
        # Returns (B, 5, D) predictions. The very last one is the state at t+1.
        return next_states[:, -1:, :]
        
    @jax.jit
    def decode(pooled):
        return probe_def.apply({'params': loaded_state['probe_params']}, pooled)

    drift_errors = []
    
    for i in tqdm(range(num_trials), desc="Stress Testing Imagination"):
        # 1. Start State
        start_joints = np.random.uniform(-1.0, 1.0, size=(6,))
        env.reset(start_joints)
        
        # Encode Start State
        img = env.render()
        img_batch = jnp.expand_dims(img, axis=0)
        latent_state = encode(img_batch) # (1, D)
        
        # 2. Initialize sliding window (Seq Length = 5)
        # We simulate the robot being stationary for 5 frames to fill the buffer
        context_buffer = jnp.repeat(jnp.expand_dims(latent_state, 1), 5, axis=1) # (1, 5, D)
        
        start_10d = forward_kinematics(np.expand_dims(start_joints, axis=0), robot_type="so100")[0]
        action_buffer = jnp.repeat(jnp.expand_dims(jnp.expand_dims(jnp.array(start_10d), 0), 1), 5, axis=1) # (1, 5, 10)
        
        # 3. Generate a smooth trajectory velocity
        action_vel = np.random.uniform(-0.1, 0.1, size=(6,))
        step_drifts = []
        
        for step in range(10):
            # True Physics Rollout
            action_joints = start_joints + action_vel * (step + 1)
            action_joints = np.clip(action_joints, -1.5, 1.5)
            action_10d = forward_kinematics(np.expand_dims(action_joints, axis=0), robot_type="so100")[0]
            
            env.step(action_joints)
            true_10d = env.get_ground_truth_10d()
            
            # AI Imagination Rollout
            # Insert newest action at the end of the sliding window
            action_expanded = jnp.expand_dims(jnp.expand_dims(jnp.array(action_10d), 0), 1) # (1, 1, 10)
            action_buffer = jnp.concatenate([action_buffer[:, 1:, :], action_expanded], axis=1) # Shift left, append right
            
            # Predict t+1
            next_latent = imagine_step(context_buffer, action_buffer) # (1, 1, D)
            
            # Decode hallucination to Cartesian coordinates
            pred_10d = decode(next_latent[0])[0]
            
            # Measure Temporal Drift
            drift = np.mean((np.array(pred_10d) - true_10d)**2)
            step_drifts.append(drift)
            
            # Append hallucinated state to the context buffer for the next autoregressive step
            context_buffer = jnp.concatenate([context_buffer[:, 1:, :], next_latent], axis=1)
            
        drift_errors.append(step_drifts)
        
    drift_errors = np.array(drift_errors) # (Trials, 10)
    avg_drifts = np.mean(drift_errors, axis=0)
    
    print("\nTemporal Drift over 10 steps:")
    for step in range(10):
        print(f"  Step {step+1}: MSE {avg_drifts[step]:.6f}")
        
    return {
        "drift_over_time": avg_drifts.tolist()
    }

if __name__ == "__main__":
    weights_path = "/home/tmainetucker/Repos/JEPA_Robotics/checkpoints/v1_jepa_backbone/v1_weights.msgpack"
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}. You must run training first.")
        sys.exit(1)
        
    env = SO100SimEnv()
    encoder_def, probe_def, wm_def, loaded_state = load_models(weights_path)
    
    metrics = {}
    metrics["perception"] = run_perception_stress_test(env, encoder_def, probe_def, loaded_state, num_trials=100)
    metrics["imagination"] = run_imagination_test(env, encoder_def, wm_def, probe_def, loaded_state, num_trials=10)
    
    with open("stress_test_report.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("\n[INFO] Saved stress_test_report.json")
