# Future Work: Boundary-Pushing Architectures

This document outlines theoretical, cross-disciplinary upgrades to the V-JEPA architecture, designed to solve fundamental limitations in current Self-Supervised Learning paradigms. 

Crucially, implementing these ideas at full scale is highly expensive. Therefore, every concept includes a **Micro-Ablation Test**. These tests are designed to isolate the mathematical theory and prove its viability in under 30 minutes of compute time before any large-scale training is attempted.

---

## 0. The Capacity Imperative: Scaling Model Size
**The Limitation:** The V1 prototype utilizes a highly constrained ~7.6M parameter model. While SMAC optimization originally selected this architecture because tiny models converge faster in short evaluation windows, 7.6M parameters is mathematically insufficient to memorize and generalize the physics of 74 highly diverse manipulation tasks (e.g., cloths, levers, hinged doors). For comparison, Meta FAIR's smallest V-JEPA 2 model is 300M parameters.
**The Solution:** Before implementing the complex mathematical theories below, Phase 2 must involve scaling the core ViT backbone to at least 50M - 100M parameters (e.g., ViT-Base). The Jetson Orin Nano possesses enough FP16 TensorRT compute overhead to maintain 30 FPS inference at this scale. Scaling the parameter count provides the geometric "memory" required for the linear probes to accurately decode complex SO(3) rotations.

---

## 1. Hyperbolic V-JEPA (Differential Geometry)
**The Limitation:** V-JEPA uses Euclidean distance (Cosine similarity), forcing the latent space onto a uniform flat hypersphere. However, a robotic arm is a hierarchical kinematic tree (shoulder ➔ elbow ➔ wrist). Hierarchical structures suffer massive distortion when embedded in Euclidean space.
**The Theory:** Project the latent representations into a Poincaré Disk (Hyperbolic space), which expands exponentially and naturally embeds hierarchical tree geometries with zero distortion.

### Micro-Ablation Test: The Bottleneck Proof
*   **Method:** Extract 1,000 pre-trained 256D latent vectors from our frozen V1 model (representing the robot arm moving). Train two tiny, 2-layer Autoencoders to compress these 256D vectors down to a severe bottleneck of just **16 dimensions**.
*   **Control:** Standard Autoencoder with Euclidean MSE loss.
*   **Test:** Autoencoder modified with a Hyperbolic distance loss (Poincaré metric) in the bottleneck.
*   **Success Metric:** If the Hyperbolic bottleneck achieves a significantly lower reconstruction error (or better Linear Probe accuracy for rotation) than the Euclidean bottleneck, it proves that hyperbolic space is fundamentally better at compressing robotic kinematics.
*   **Compute Time:** ~5 minutes.

---

## 2. Entropy-Guided Masking (Thermodynamics)
**The Limitation:** Randomly masking 75% of an image is information-theoretically inefficient. Masking a blank wall teaches the network nothing; masking the robotic gripper teaches it everything.
**The Theory:** Use a spatial-frequency filter (Sobel edge detection) to calculate the "informational entropy" of the image. Dynamically force the masks onto the highest-entropy patches, forcing the model to only solve the most complex physical puzzles.

### Micro-Ablation Test: The 5-Epoch Sprint
*   **Method:** Create a micro-dataset of 1,000 highly diverse frames. Initialize two identical, tiny ViT models (e.g., Depth 2).
*   **Control:** Train for 5 epochs using standard 75% random masking.
*   **Test:** Train for 5 epochs using Entropy-Guided masking (targeting the highest-variance patches).
*   **Success Metric:** Measure the `Val Avg L` and Linear Probe scores after exactly 5 epochs. If the Test model achieves a lower validation loss significantly faster, it proves massive sample-efficiency gains.
*   **Compute Time:** ~20 minutes.

---

## 3. Hamiltonian World Models (Classical Mechanics)
**The Limitation:** The World Model uses a Transformer to predict $S_{t+1}$. Transformers process time symmetrically and do not inherently understand the causal "Arrow of Time" or conservation of energy, allowing them to accidentally predict physically impossible futures (hallucinations).
**The Theory:** Force the World Model to obey Hamiltonian mechanics. Instead of predicting the next state, it predicts the *gradients of a Latent Energy function* (splitting the latent vector into Position and Momentum), guaranteeing the predictions obey conservation laws.

### Micro-Ablation Test: The Pendulum Rollout
*   **Method:** Disconnect the vision system entirely. Generate a simple, synthetic 1D physics dataset (a simulated swinging pendulum defined by `[angle, velocity]`).
*   **Control:** Train a standard MLP to predict state $T+1$ from state $T$.
*   **Test:** Train a Hamiltonian Neural Network (HNN) to predict the energy gradients, deriving $T+1$ via Euler integration.
*   **Success Metric:** Roll out the predictions without ground-truth corrections for 100 timesteps. The standard MLP will slowly drift, gaining or losing energy (the pendulum will spin out of control or stop). The HNN will perfectly conserve energy, maintaining a flawless orbit.
*   **Compute Time:** ~10 minutes.

---

## 4. Orthogonal Gradient Descent (Neuroscience)
**The Limitation:** Catastrophic Forgetting. If we train the robot to grasp a cup (Task A) and then train it to fold laundry (Task B), backpropagation will overwrite the cup-picking weights, causing the robot to forget Task A.
**The Theory:** When transitioning to Task B, compute the Null Space (the orthogonal subspace) of the gradients from Task A. Force the optimizer to only update weights in this orthogonal direction, ensuring the knowledge of Task A is mathematically protected.

### Micro-Ablation Test: The Split-Task Proof
*   **Method:** Ignore V-JEPA. Use a simple dataset like Split-MNIST or CIFAR-10. Train a basic CNN on Task A (e.g., identifying digits 0-4).
*   **Control:** Fine-tune the CNN on Task B (digits 5-9) using standard AdamW.
*   **Test:** Fine-tune the CNN on Task B using Orthogonal Gradient Descent (projecting updates into the Null Space of Task A's Fisher Information Matrix).
*   **Success Metric:** Measure accuracy on Task A after training on Task B. The Control will crash to ~0% accuracy on Task A. If the Test model maintains >95% accuracy on Task A while successfully learning Task B, the mathematics are verified for large-scale scaling.
*   **Compute Time:** ~15 minutes.

---

## 5. Physics-Informed Policy Head (Safe Reinforcement Learning)
**The Limitation:** Standard Behavioral Cloning uses MSE loss to train the Policy Head to match human demonstrations. The neural network has no inherent knowledge of the robot's physical body, meaning it can hallucinate and output mathematically impossible or hardware-damaging actions (e.g., smashing the gripper through the table, exceeding joint limits, or producing jagged movements that burn out motors).
**The Theory:** Inject Physics-Informed Machine Learning (PIML) directly into the Policy Head's loss function. We apply severe penalties for actions that violate the robot's kinematic singularity limits (maximum reach radius), table-surface collisions ($Z < 0$), and high "Jerk" (the 3rd derivative of position). This forces the Policy Head to strictly output physically realizable, motor-safe trajectories.

### Micro-Ablation Test: The Z-Floor Collision Avoidance
*   **Method:** Generate a synthetic 3D trajectory dataset of a point moving around a box, but randomly inject 10% corrupted data points where the coordinate goes below the table ($Z < -0.5$). Train two simple MLP Policy Heads to predict the coordinates.
*   **Control:** Train with standard MSE loss.
*   **Test:** Train with Physics-Informed Loss (MSE + severe penalty if predicted $Z < 0$).
*   **Success Metric:** Run inference on the trained models and measure the lowest predicted Z value. The Control model will occasionally predict $Z < 0$ (smashing the table) because it learned from the corrupted data. The Test model will learn a strict safety boundary and never predict $Z < 0$, proving that hard-coded physics constraints mathematically override corrupted training data.
*   **Compute Time:** ~3 minutes.

---

## 6. The MuJoCo Latent Physics Consistency Test (Simulation Verification)
**The Limitation:** Currently, we measure the World Model's accuracy using mathematical loss (Cosine MSE) against future frames. However, mathematical loss does not tell us if the network's "hallucinations" strictly obey the physical laws of the universe (e.g., conservation of momentum, gravity, collision solids). A low MSE loss could still allow the robot arm to briefly phase through a table.
**The Theory:** Instead of changing the neural network, we use a deterministic physics engine (MuJoCo) as an external algorithmic "judge" to grade the World Model's physical understanding. We force the World Model to dream a 50-step trajectory, decode those latents into coordinates, and feed them into MuJoCo to check for physical impossibilities.

### Micro-Ablation Test: The Hallucination Audit
*   **Method:** Provide the frozen V1 World Model with a single initial image frame ($S_0$) and a sequence of 50 actions. Run the World Model forward to generate 50 hallucinated future latents ($S_1 \dots S_{50}$). Decode these latents using the Linear Probe into Cartesian coordinates, and load them into a `mujoco_assets` simulation environment.
*   **Control:** Standard mathematical MSE loss evaluation on the latents.
*   **Test:** MuJoCo Physics Audit. The simulator explicitly checks the trajectory for three strict violations: 
    1. **Teleportation:** (Velocity/Energy limit exceeded between frames).
    2. **Ghosting:** (Solid bodies overlapping, violating Pauli exclusion principles).
    3. **Floating:** (Objects hovering without upward force, violating gravity).
*   **Success Metric:** If the 50-step dreamed trajectory passes the MuJoCo simulation with 0 physics violations, it mathematically proves that the self-supervised architecture has successfully internalized classical mechanics purely from pixel observation.
*   **Compute Time:** ~1 minute (Standard forward pass + MuJoCo physics step).
