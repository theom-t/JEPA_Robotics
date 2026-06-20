# Cross-Embodiment Latent World Model: R&D V1 Plan

## 1. Core Hypothesis
We hypothesize that a **Joint-Embedding Predictive Architecture (JEPA)** trained via Cross-Embodiment Learning can learn robust, zero-shot physical dynamics. By fusing generalist robotic datasets (BridgeData V2) with localized teleoperation data (LeRobot SO100) via a standardized 7D Cartesian kinematics bridge, the V-JEPA Perception Engine will compress raw video feeds into an abstract latent state. 

From this latent state, an **Action-Conditioned Transformer World Model** can simulate future physical states, allowing for optimal trajectory planning at the edge (Jetson Orin Nano).

## 2. Dynamic Optimization Paradigm
Rather than hardcoding arbitrary network dimensions (which leads to sub-optimal edge inference or intelligence bottlenecks), this architecture is parameterized. We utilize **SMAC3** (Sequential Model-based Algorithm Configuration) to explore a complex, hierarchical search space.

### SMAC3 Hierarchical Search Space
If the World Model is initialized as a Transformer, SMAC3 will optimize:
- `latent_dim`: The dense representation size output by the V-JEPA encoder.
- `transformer_depth`: The number of self-attention blocks in the World Model.
- `transformer_heads`: The number of attention heads (constrained to factor cleanly into `latent_dim`).
- `tau`: The Exponential Moving Average (EMA) momentum for the V-JEPA Target Encoder.

## 3. Architecture Definitions

### 3.1. V-JEPA Perception Engine
- **Context Encoder ($E_x$)**: A Vision Transformer (ViT) module operating on masked image patches.
- **Target Encoder ($E_y$)**: An exact structural clone of $E_x$. Its weights are not updated via backpropagation, but via an EMA of the Context Encoder to prevent representation collapse.
- **Predictor ($P$)**: A lightweight neural network predicting the latent state of unmasked target patches.

### 3.2. Action-Conditioned Transformer (Latent World Model)
A temporal simulation engine taking a sequence of historical states $[s_{t-k}, \dots, s_t]$ and planned actions $[a_{t-k}, \dots, a_t]$. It utilizes causal multi-head self-attention to predict the immediate next physical state $\hat{s}_{t+1}$. This mechanism forces the model to encode temporal dynamics, momentum, and gravity into its predictions.

## 4. Telemetry & Validation Protocols
To guarantee mathematical safety and provide diagnostic visibility during SMAC3 optimization, the architecture is instrumented with the following validation metrics:

### Stage 1: Perception (Latent L2 Loss)
We calculate the L2 distance between the Predicted Latent State and the True Target Latent State ($\| \hat{s}_y - s_y \|_2^2$). 
*Validation:* Ensures the ViT is accurately learning physical abstractions without relying on pixel-perfect reconstruction.

### Stage 2: Temporal Dynamics Loss
We validate the output of the World Model ($\hat{s}_{t+1}$) against the actual encoded next frame.
*Validation:* Ensures the Action-Conditioned Transformer isn't cheating (causal masking) and successfully understands kinematics.

### Stage 3: The Pareto Optimization Front
SMAC3 evaluates the end-to-end architecture on a multi-objective Pareto front:
1. Minimizing the combined JEPA L2 + Temporal Dynamics losses.
2. Minimizing FLOPs/Inference Latency to guarantee 30Hz+ operational frequencies on the target edge hardware (Jetson Orin Nano).
