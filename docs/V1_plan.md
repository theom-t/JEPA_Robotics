# Cross-Embodiment Latent World Model: R&D V1 Plan

## 1. Core Hypothesis
We hypothesize that a **Joint-Embedding Predictive Architecture (V-JEPA)** trained via Cross-Embodiment Learning can learn robust, zero-shot physical dynamics. By fusing massive generalist robotic datasets (Google BridgeData V2) with localized teleoperation data (LeRobot SO100) via a standardized 7D Cartesian kinematics bridge, the V-JEPA Perception Engine will compress raw video feeds into an abstract latent state. 

From this latent state, an **Action-Conditioned Transformer World Model** simulates future physical states, allowing for optimal trajectory planning at the edge (Jetson Orin Nano).

## 2. Dynamic Optimization Paradigm (SMAC3 Hyperband)
Rather than hardcoding network dimensions, this architecture is parametrically optimized using **SMAC3 (Sequential Model-based Algorithm Configuration)**.

To evaluate dozens of architectures efficiently, we employ a **Multi-Fidelity Hyperband Intensifier**:
- **Epoch Budgeting:** Architectures are initially given a minimum budget of 4 epochs. (We wait until Epoch 4 to avoid prematurely pruning architectures during the "JEPA bump" where latent representation spaces physically expand).
- **Successive Halving:** After 4 epochs, the worst 50% (`eta=2`) of configurations are pruned. The survivors are promoted to 8 epochs, and finally to a maximum budget of 10 epochs.
- **Stratified Subsets:** During optimization, SMAC only trains on a perfectly distributed `2%` slice of the datasets (capped at 500 batches per epoch) to evaluate configurations in ~1.5 minutes each.
- **Validation Objective:** SMAC optimizes strictly for **Validation Loss**, inherently penalizing architectures that memorize rather than generalize.

## 3. Decoupled Data Ingestion & Pipelines
Due to severe dependency conflicts between JAX/Blackwell libraries and Google Protocol Buffers, the system uses a strictly decoupled environment architecture:

- **`jepa_data` Environment:** Used exclusively for ingestion. It bypasses `tensorflow-datasets` entirely, utilizing direct `gsutil -m rsync -c` to stream raw RLDS TFRecords from Google Cloud. HuggingFace datasets are cached using forced offline limits (`HF_DATASETS_OFFLINE=1`).
- **Cross-Embodiment Alignment (`jepa_robotics`):** The JAX dataloaders center-crop videos to `[256, 256, 3]` and slide them into temporal sequences of `seq_len=5`. 
- **Infinite Cyclic Generation:** To prevent the massive BridgeData dataset (105,000 frames) from being truncated by the smaller SO100 dataset (22,000 frames), the data pipeline uses a `cycle_loader` generator. This infinitely loops the smaller dataset so cross-embodiment batches are flawlessly zipped together.

## 4. Architecture Definitions

### 4.1. V-JEPA Perception Engine
- **Context Encoder ($E_x$)**: A Vision Transformer (ViT) module operating on masked image patches.
- **Target Encoder ($E_y$)**: An exact structural clone of $E_x$. Its weights are not updated via backpropagation, but via an EMA (`tau`) of the Context Encoder to prevent representation collapse.
- **Predictor ($P$)**: A lightweight neural network predicting the latent state of unmasked target patches.

### 4.2. Action-Conditioned Transformer (Latent World Model)
A temporal simulation engine taking a sequence of historical states $[s_{t-k}, \dots, s_t]$ and planned actions $[a_{t-k}, \dots, a_t]$. It utilizes causal multi-head self-attention to predict the immediate next physical state $\hat{s}_{t+1}$. 

### 4.3. Adaptive Evasive Maneuvers (Dynamic Target Freezing)
A static EMA update schedule (`tau=0.995`) is extremely brittle over long 100-epoch runs, as it allows the Target Network to be slowly dragged into a zero-variance "Constant-State" collapse if the Context Network discovers a mathematical loophole in the Cosine Distance loss. Conversely, a permanently frozen target (`tau=1.0`) prevents collapse but yields a Latent space with weaker semantic grouping. 

To achieve the best of both worlds, the architecture employs an **Adaptive Evasive Maneuver**:
- The training loop maintains an Exponential Moving Average of the `sig_reg` variance penalty (an InfoMax metric tracking the variance of the latents).
- **🟢 EMA (Healthy State):** When the `sig_reg` penalty is low, the network is expanding into the full Latent space. The Target Network smoothly updates via EMA (`tau=0.995`), ensuring deep semantic grouping.
- **🥶 FROZEN (Evasive Maneuver):** If the Context Network begins to collapse, the `sig_reg` penalty spikes. The training loop intercepts this and instantly hard-freezes the Target Network (`current_tau = 1.0`). The Context Network is then violently forced by the variance penalty to expand out of the collapse zone, but cannot drag the frozen Target down with it. Once healthy variance is restored, the EMA schedule seamlessly resumes.

## 5. Telemetry & Validation Protocols
The architecture is instrumented with the following validation metrics:

### Stage 1: Perception (Latent L2 Loss)
The L2 distance between the Predicted Latent State and the True Target Latent State ($\| \hat{s}_y - s_y \|_2^2$). 
*Validation:* Ensures the ViT is accurately learning physical abstractions without relying on pixel-perfect reconstruction.

### Stage 2: Temporal Dynamics Loss
Validates the output of the World Model ($\hat{s}_{t+1}$) against the actual encoded next frame.
*Validation:* Ensures the Action-Conditioned Transformer successfully understands kinematics.

### Stage 3: Human-Interpretable Kinematics (Linear Probe)
A `StateLinearProbe` continuously attempts to decode the 7D Cartesian XYZ/Roll-Pitch-Yaw coordinates from the abstract V-JEPA latent space. 
*Validation:* Because self-supervised L2 losses are uncalibrated and prone to "representation collapse", the **SMAC objective uses a Disentangled Weighted Probe Score**. The 7D metric is split into Position (XYZ), Rotation (Roll-Pitch-Yaw), and Gripper errors. SMAC optimizes a weighted sum (`Pos*1.0 + Rot*0.1 + Grp*0.1`) to ensure large rotational radian errors do not mathematically crush tiny positional meter errors.

### Stage 4: Blackwell (RTX 5090) Hardening
The codebase is hardened against XLA C++ autotuning spam and TPU initialization crashes using strict environment overrides (`TF_CPP_MIN_LOG_LEVEL=3`, `JAX_PLATFORMS=cuda,cpu`), ensuring clean telemetry generation.

### Stage 5: Virtual Sandbox Validation
Before deploying the V1 weights to a physical robot or advancing to V2, the system must pass a dual-tier virtual evaluation using a Mujoco physics simulation of the LeRobot SO100 arm. Because V1 is a state estimator and world model (not a policy), these tests evaluate *perception* and *imagination*, not task completion.

#### 1. Automated Headless Testing (The "Stress Tester")
A Python script runs thousands of randomized virtual trajectories headlessly in Mujoco:
* **Perception Stress Test:** The simulator renders an image of the arm. The ViT + Linear Probe predicts the 10D pose coordinates. The camera is then virtually perturbed (moved slightly out of the training distribution) to generate a degradation curve, proving whether the ViT learned true 3D physics or merely memorized pixel patterns.
* **Imagination Test:** The World Model is given a starting image and 10 future actions. It imagines the next 10 latent states, which the probe decodes into 10D coordinates. These are compared against the true Mujoco physics simulation to verify temporal forecasting accuracy.

#### 2. Visual Interactive Testing (The "AI Debugger")
A live, interactive visualizer designed to run on the home Linux host and stream over SSH to the remote laptop.
* **Architecture:** To support smooth remote SSH viewing, the visualizer uses a lightweight Flask/FastAPI server to stream a high-framerate MJPEG feed of the Mujoco render directly to the remote laptop's web browser.
* **Interactive Evaluation:** The user can manually drag the virtual SO100 arm around on the screen. A "Ghost Arm" overlay, driven entirely by the ViT's real-time coordinate predictions, must perfectly track the user's movements. If the user clicks "Imagine Trajectory", the World Model will cast a holographic prediction of the arm's future path, allowing intuitive, visual debugging of the AI's internal physics engine.
