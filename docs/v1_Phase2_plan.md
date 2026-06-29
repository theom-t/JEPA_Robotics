# V1 Phase 2: Next-Generation Pre-Training Plan

If we ever need to re-train the V1 backbone from scratch, we will implement this "Phase 2" plan. This plan incorporates all the empirical telemetry findings from our first 100-epoch run, as well as the architectural upgrades discovered by Meta FAIR (V-JEPA 2.1).

## 1. Architectural Upgrades (V-JEPA 2.1 Integration)
To achieve state-of-the-art dense features right out of the gate, we will modify the core architecture before Epoch 0:

*   **Dense Predictive Loss:** Instead of only predicting the masked (invisible) patches, the `JEPAPredictor` and the `loss` function will be modified to predict *all* tokens (both masked and visible context patches). This forces the network to learn ultra-consistent, high-quality dense features across the entire image.
*   **Deep Self-Supervision:** We will extract intermediate activations from both the Context Predictor and the Target Encoder (e.g., at Layer 2 and Layer 4) and apply the loss function at multiple depths. This prevents the early attention blocks from being "lazy" and forces them to learn rich geometric structure immediately.

## 2. Telemetry & Regularization Discoveries
Our first run provided profound insights into how self-supervised Vision Transformers behave under a Cosine Learning Rate Schedule. We will adhere to these rules:

*   **The "Bouncy Castle" Optimum:** We found that `sigreg_weight = 0.02` with a pure Cosine EMA decay (`tau = 0.995 -> 1.0`) is the mathematical optimum. The variance (`SReg`) will gently bounce between `0.5` and `1.2`. 
*   **No Evasive Maneuvers:** We absolutely will **not** use adaptive threshold freezing (freezing the target network if variance drops). This causes a deadlock. The `0.02` repulsive force is enough to guarantee immunity from zero-variance collapse.
*   **The Illusion of Plateau:** Validation metrics (`Pos`, `Rot`, `Grp`) will appear to plateau heavily between Epoch 30 and Epoch 70. This is **normal**. Because the learning rate is following a Cosine curve, it is still at ~40% of its peak at Epoch 60. The network is actively exploring the latent space and cannot settle. The true performance is only revealed during the final cooldown phase (Epoch 80+).
*   **Linear Probe Bottlenecks:** We accept that a simple Linear Probe (`Dense(10)`) fundamentally underestimates the network's true physical understanding because it cannot decode non-linear geometric manifolds. The probe is just a baseline diagnostic, not the ceiling of the model's intelligence.

## 3. The 70/30 Curriculum Learning Strategy
We will purposefully use a Curriculum Learning strategy to maximize environmental diversity without causing catastrophic forgetting during the high-learning-rate phase.

1.  **Phase A (Epochs 0 - 70): The Exploration Phase**
    *   Run with `--fraction 0.5` (50% dataset).
    *   The optimizer is taking massive steps (Cosine LR is high).
    *   The smaller dataset allows the network to rapidly build core macro-physics (object permanence, gravity, basic XYZ coordinates) very quickly.
2.  **Phase B (Epochs 71 - 100): The 100% Data Fine-Tuning Phase**
    *   Stop the script at Epoch 70 and restart with `--fraction 1.0`.
    *   At Epoch 70, the Cosine Learning Rate drops into the universally accepted "Fine-Tuning" range (~20% of peak, `~6e-5`).
    *   We suddenly flood the network with double the environmental diversity, camera angles, and lighting conditions. 
    *   Because the learning rate is low, the network gently carves out space for these new concepts without violently unlearning the macro-physics it built in Phase A.

## 4. Post-Training (V1.5 High-Acuity Burst)
Once the 100-epoch Phase 2 run concludes at `256x256`, we will still execute the V1.5 High-Acuity Burst. 
*   Interpolate the positional embeddings from 16x16 to 32x32.
*   Increase dataloader resolution to `512x512` (1024 patches).
*   Fine-tune for 20 epochs with a smaller batch size to drastically sharpen `Rot` and `Grp` metrics before freezing the model for V2 Policy Head training.
