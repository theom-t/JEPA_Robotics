## Project Proposal: Cross-Embodiment Latent World Model for Edge Robotics
**Objective:** To engineer and deploy a Joint-Embedding Predictive Architecture (JEPA) and latent World Model capable of zero-shot physical reasoning. The system will leverage Cross-Embodiment Learning to train on heterogeneous datasets (BridgeData V2 and LeRobot SO100) before being deployed to a localized edge-compute environment controlling a low-cost, 6-DoF robotic arm.
### 1. Hardware & Compute Architecture
The system pipeline is strictly bifurcated into a high-compute training environment and an optimized edge-inference environment.
 * **Training & Simulation Node:** Localized Linux desktop equipped with an NVIDIA RTX 5090 (32GB VRAM). This handles the high-memory requirements of the dual-network (EMA) JEPA architecture and large-batch multithreaded data loading.
 * **Edge Inference Node:** NVIDIA Jetson Orin Nano. Handles real-time video ingestion, latent state encoding, optimal trajectory planning, and Inverse Kinematics (IK) calculation.
 * **Physical Hardware:** 3D-printed LeRobot SO100 6-DoF robotic arm (approximate BOM: $150-$200), utilizing standard PWM servo motors.
### 2. Data Strategy: Cross-Embodiment Alignment
To bypass the Sim-to-Real gap while avoiding the costs of industrial hardware, the training pipeline will fuse two distinct real-world datasets.
 * **Dataset A (The Generalist Prior):** BridgeData V2 (WidowX arm). Provides 60,000+ trajectories across 24 environments. This forces the perception encoder to learn robust, noise-invariant physical mechanics.
 * **Dataset B (The Embodiment Anchor):** LeRobot SO100 teleoperation data. Provides specific visual and kinematic grounding for the target 3D-printed hardware.
**The Kinematic Bridge:** To make these datasets mathematically compatible for a single neural network, all joint-space motor logs will be converted into Cartesian Task Space. Both datasets will represent actions exclusively as 7-dimensional end-effector vectors: [\Delta X, \Delta Y, \Delta Z, \Delta\text{roll}, \Delta\text{pitch}, \Delta\text{yaw}, \text{gripper\_state}].
### 3. Model Architecture
The software stack moves away from standard generative transformers, utilizing a model-based reasoning approach defined by two core engines.
#### The Perception Engine (V-JEPA)
A Vision-JEPA model processes live, multi-angle camera feeds. Instead of predicting raw pixels, it maps the physical environment into a low-dimensional abstract state.
 * **Context Encoder (E_x):** Mapped via gradient descent.
 * **Target Encoder (E_y):** Updated via Exponential Moving Average (EMA) to prevent representation collapse.
 * **Loss Function:** Optimized in the latent space, e.g., \mathcal{L} = \| \hat{s}_y - s_y \|_2^2, filtering out unpredictable visual noise.
#### The Reasoning Engine (Latent World Model)
An action-conditioned predictor that simulates future states entirely within the abstract latent space.
 * **Simulation Loop:** Given a latent state s_t and a proposed action sequence [a_t, a_{t+1}, \dots], the model recursively predicts future states [\hat{s}_{t+1}, \hat{s}_{t+2}, \dots].
 * **Planning:** Evaluates predicted trajectories against a target latent state, backpropagating through time to find the optimal Cartesian action vector.
### 4. Execution Roadmap
#### Phase I: Data Unification and Pre-training
 1. **Ingestion:** Utilize the Hugging Face lerobot library to stream both BridgeData V2 (via RLDS wrappers) and SO100 Parquet/MP4 files into a unified PyTorch or JAX dataloader.
 2. **Kinematic Transformation:** Apply Forward Kinematics (FK) to ensure all target action labels are formatted as uniform 7D Cartesian vectors.
 3. **VRAM Optimization:** Train the V-JEPA perception engine on the RTX 5090 using mixed-precision training and gradient checkpointing to manage the heavy memory footprint of the video transformer.
#### Phase II: Edge Optimization
Migrating a heavily parameterized JEPA encoder and World Model onto the Jetson Orin Nano requires aggressive compression to maintain real-time control frequencies (e.g., 30Hz+ inference).
 1. **Automated Quantization:** Reduce the precision of the frozen model weights (e.g., FP32 to INT8) using post-training quantization, calibrating against a subset of the SO100 dataset to ensure the latent representation space does not degrade.
 2. **Optimized Pruning:** Implement automated pruning pipelines to strip redundant attention heads and sparse weight connections in the perception encoder, directly reducing the FLOPs required per frame on the Orin Nano.
#### Phase III: Hardware Deployment and Control
 1. **Physical Assembly:** Print and assemble the SO100 chassis and flash the servo control firmware.
 2. **The Inference Loop:** The Orin Nano ingests the camera feed, passes it through the quantized V-JEPA encoder, and queries the World Model for the optimal Cartesian action (\Delta X, \Delta Y, \Delta Z).
 3. **Local IK Execution:** A local Inverse Kinematics solver translates the commanded Cartesian task-space coordinates into specific PWM signals for the SO100 servo motors.
 4. **Few-Shot Fine-Tuning:** Collect 50-100 local teleoperated demonstrations to fine-tune the final action-readout layer, calibrating the model to the exact motor backlash and friction of the printed chassis.
