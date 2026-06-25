# JEPA Robotics - System Architecture

This document tracks the high-level software architecture for the Cross-Embodiment Latent World Model.

## 1. Data Ingestion & Preprocessing
*   **Path:** `src/jepa_robotics/data/`
*   **Design Pattern:** Object-Oriented inheritance model. `BaseRobotDataset` acts as the contract for all robot loaders.
*   **Implementations:**
    *   `BridgeDataLoader`: WidowX 250 data.
    *   `SO100DataLoader`: LeRobot SO100 data.
*   **Goal:** Convert varying inputs (Hugging Face datasets, MP4s, Parquet files, TFRecords) into uniform JAX batches: `(Images, Joint States, Joint Actions)`.

## 2. Kinematics Engine
*   **Path:** `src/jepa_robotics/data/kinematics.py`
*   **Core Function:** Maps arbitrary Degrees of Freedom (DoF) to a unified **7D Cartesian State Space** `[X, Y, Z, Roll, Pitch, Yaw, Gripper_State]`.
*   **Technology:** JAX-accelerated matrix multiplications (Forward Kinematics based on Denavit-Hartenberg parameters).

## 3. Perception Engine (V-JEPA)
*   **Goal:** Compress raw pixel data into a dense, informative latent space without relying on pixel reconstruction.
*   **Architecture:** Vision Transformer (ViT) implemented via `flax.linen`.
*   **True Spatial Masking:** Each training step generates a random spatial mask splitting the `N_patches` into:
    *   **Context patches** (`N_context = N * (1 - masking_ratio)`): passed to the Context Encoder (E_x) via `patch_indices`. E_x is completely blind to the masked region.
    *   **Target patches** (`N_target = N * masking_ratio`): held out from E_x.
    *   Positional embeddings are applied to ALL patches *before* selection, preserving spatial coordinates in the context tokens.
*   **Context Encoder (E_x):** Processes only `N_context` patch tokens. Updated via backpropagation.
*   **Target Encoder (E_y):** Processes all `N_patches` (no masking). Updated via EMA (`tau`) of E_x — never by backprop. Provides ground-truth latent targets.
*   **Predictor (P):** Receives E_x context latents + target positional embeddings (from E_y's param table). A learnable mask token is placed at each target position and attended over jointly with context latents. Outputs per-patch predictions for the masked region only.
*   **JEPA Loss:** L2 distance computed **only on the masked patch positions** in latent space — no pixel reconstruction.
*   **Modularity:** The encoder can be frozen, quantized, and deployed to edge hardware (Jetson Orin Nano) independently of the World Model.

## 4. Latent World Model
*   **Goal:** Predict future states and plan actions entirely within the latent space derived by the V-JEPA encoder, conditioned on the 7D Cartesian action space.
*   **Architecture:** Action-Conditioned Transformer. Utilizing causal multi-head self-attention, it predicts $\hat{s}_{t+1}$ using a temporal sequence of historical latents and actions.
*   **Hyperparameter Initialization:** Transformer layers and attention heads are parameterized for SMAC3 optimization.

## 5. Hyperparameter Optimization (SMAC3)
*   **Goal:** Navigate the highly complex, hierarchical search space of our dual-engine architecture.
*   **Validation Pareto Front:** SMAC3 continuously balances the models against a dual-objective metric: minimizing Latent/Temporal loss while simultaneously minimizing Forward Pass Latency to ensure it meets edge robotics bounds (30Hz+).

## 5. Hardware & Environment
*   **Training:** Local NVIDIA RTX 5090 (Blackwell architecture). Requires JAX `0.6.2` via `pip` (CUDA 12.8 compatible).
*   **Edge Inference:** NVIDIA Jetson Orin Nano (NVMe boot).

## 6. Testing & Verification Protocol
*   **Framework:** `pytest` is the mandated testing framework for all modules.
*   **Directory:** All tests reside in the `tests/` directory mirroring the `src/` layout.
*   **Coverage:** `pytest-cov` should be used to monitor coverage on JAX tensor operations and mathematical boundaries.
