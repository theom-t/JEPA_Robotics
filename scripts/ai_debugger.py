import os
import sys
# Suppress XLA C++ warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_PLATFORMS"] = "cuda,cpu"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import time
import cv2
import threading
import numpy as np
from flask import Flask, render_template, Response, request, jsonify

# JAX & JEPA Imports
import jax
import jax.numpy as jnp
from jepa_robotics.models.v_jepa import ViTEncoder, StateLinearProbe
from jepa_robotics.evaluation.mujoco_env import SO100SimEnv, SO100GhostEnv
from jepa_robotics.evaluation.ik_solver import IKSolver
from jepa_robotics.models.world_model import ActionConditionedTransformer
from jepa_robotics.data.kinematics import forward_kinematics
import flax.serialization

app = Flask(__name__, template_folder='../templates')

# Global State
weights_loaded = False
env = None
ghost_env = None
ik_solver = None
encoder_def = None
probe_def = None
wm_def = None
loaded_state = None

# We keep track of the user's dragged joint angles
current_joints = np.zeros(6)
context_buffer = None
action_buffer = None
imagination_trigger = False

@jax.jit
def perceive(img):
    _, pooled = encoder_def.apply({'params': loaded_state['encoder_params']['params']}, img, patch_indices=None)
    pred_10d = probe_def.apply({'params': loaded_state['probe_params']['params']}, pooled)
    return pooled, pred_10d

@jax.jit
def imagine_step(seq_context, seq_actions):
    next_states = wm_def.apply({'params': loaded_state['wm_params']['params']}, seq_context, seq_actions)
    return next_states[:, -1:, :]

latest_frame = None

def render_loop():
    global current_joints, context_buffer, action_buffer, imagination_trigger, latest_frame
    global env, ghost_env, ik_solver
    
    print("[INFO] Initializing MuJoCo Environments in Background Thread...")
    env = SO100SimEnv()
    ghost_env = SO100GhostEnv()
    ik_solver = IKSolver()
    
    while True:
        if not weights_loaded:
            time.sleep(1)
            continue
            
        # 1. Update true physics to match user drag
        env.reset(current_joints)
        
        # 2. Render true environment
        real_img_float = env.render()
        
        # 3. Perceive 10D State with V-JEPA
        img_batch = jnp.expand_dims(real_img_float, axis=0)
        latent_state, pred_10d = perceive(img_batch)
        latent_state = jnp.expand_dims(latent_state, axis=1) # (1, 1, D)
        pred_10d = pred_10d[0]
        
        # Initialize or update the 5-frame rolling buffers
        true_10d = forward_kinematics(np.expand_dims(current_joints, axis=0), robot_type="so100")[0]
        action_expanded = jnp.expand_dims(jnp.expand_dims(jnp.array(true_10d), 0), 1)
        
        if context_buffer is None:
            context_buffer = jnp.repeat(latent_state, 5, axis=1)
            action_buffer = jnp.repeat(action_expanded, 5, axis=1)
        else:
            context_buffer = jnp.concatenate([context_buffer[:, 1:, :], latent_state], axis=1)
            action_buffer = jnp.concatenate([action_buffer[:, 1:, :], action_expanded], axis=1)
            
        if imagination_trigger:
            sim_context = context_buffer
            sim_action = action_buffer
            # We imagine a smooth pan of the shoulder
            imagined_joints = current_joints.copy()
            for step in range(10):
                imagined_joints[0] += 0.1 # Pan left
                img_action_10d = forward_kinematics(np.expand_dims(imagined_joints, axis=0), robot_type="so100")[0]
                img_action_expanded = jnp.expand_dims(jnp.expand_dims(jnp.array(img_action_10d), 0), 1)
                sim_action = jnp.concatenate([sim_action[:, 1:, :], img_action_expanded], axis=1)
                
                next_latent = imagine_step(sim_context, sim_action)
                pred_ghost_10d = probe_def.apply({'params': loaded_state['probe_params']['params']}, next_latent[0])[0]
                
                ghost_joints = ik_solver.solve(np.array(pred_ghost_10d), initial_guess=imagined_joints)
                
                # Render just this single imagined step
                blended_step = ghost_env.overlay_ghost(real_img_float, ghost_joints)
                
                # Instantly encode and publish the frame to create an animation!
                bgr_step = cv2.cvtColor((blended_step * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                ret_step, buffer_step = cv2.imencode('.jpg', bgr_step, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                latest_frame = (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer_step.tobytes() + b'\r\n')
                
                # Advance context
                sim_context = jnp.concatenate([sim_context[:, 1:, :], next_latent], axis=1)
                time.sleep(0.1) # 10fps playback
                
            imagination_trigger = False
            time.sleep(1.0) # Pause at the end for 1s so the user can see it
            continue
        else:
            # Standard Real-time Ghost
            ghost_joints = ik_solver.solve(np.array(pred_10d), initial_guess=current_joints)
            blended_float = ghost_env.overlay_ghost(real_img_float, ghost_joints)
        
        # Convert to BGR uint8 for JPEG encoding
        bgr = cv2.cvtColor((blended_float * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Encode
        ret, buffer = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame = buffer.tobytes()
        
        latest_frame = (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.01)

def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            yield latest_frame
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_joints', methods=['POST'])
def update_joints():
    global current_joints
    data = request.json
    current_joints = np.array(data['joints'])
    return jsonify({"status": "ok"})

@app.route('/imagine', methods=['POST'])
def imagine_trajectory():
    global imagination_trigger
    imagination_trigger = True
    return jsonify({"status": "ok"})

def start_server():
    global encoder_def, probe_def, wm_def, loaded_state, weights_loaded
    
    weights_path = "/home/tmainetucker/Repos/JEPA_Robotics/checkpoints/v1_jepa_backbone/v1_weights.msgpack"
    if os.path.exists(weights_path):
        print("[INFO] Loading V-JEPA Weights...")
        encoder_def = ViTEncoder(latent_dim=256, depth=4, num_heads=16, patch_size=16, activation_fn="gelu")
        probe_def = StateLinearProbe(out_dim=10)
        wm_def = ActionConditionedTransformer(latent_dim=256, depth=4, num_heads=16, activation_fn="gelu")
        
        with open(weights_path, "rb") as f:
            loaded_state = flax.serialization.msgpack_restore(f.read())
            
        weights_loaded = True
        print("[INFO] Ready! Ghost Tracker active. Open http://localhost:5000 in your browser.")
    else:
        print(f"[WARNING] {weights_path} not found. Waiting for training to finish...")
        print("[INFO] Server will start, but MJPEG stream will hang until weights are available.")
    t = threading.Thread(target=render_loop, daemon=True)
    t.start()
        
    app.run(host='0.0.0.0', port=5000, threaded=True)

if __name__ == '__main__':
    start_server()
