# V2 Plan: Language-Conditioned Autonomous Stationary Arm

This document outlines the architectural roadmap for transitioning the V1 Cross-Embodiment JEPA (Self-Supervised Latent World Model) into a V2 Autonomous Agent. 

V2 specifically focuses on deploying the pre-trained V1 Vision Encoder onto a **stationary robotic arm** (e.g., LeRobot SO100 or WidowX 250 bolted to a table) controlled via a local **Jetson Orin Nano**. The system will execute language-conditioned tasks using an action-predicting Policy Head.

*(Note: V3 will tackle the complexities of mobile manipulation, egocentric wrist-cameras, and full room navigation).*

---

## 1. The Language Interface (Cross-Attention)
To command the robot to perform specific semantic tasks, the pipeline must be multimodal.

* **Objective:** Inject language understanding into the visual latent space.
* **Architecture Upgrade:** 
    * Integrate a frozen, lightweight Text Encoder (e.g., OpenAI CLIP Text or a small LLM like T5).
    * Feed user commands (e.g., *"Pick up the sponge"*) into the Text Encoder to extract a dense text embedding.
    * Inject the Text Embedding into the ViT Encoder using a **Cross-Attention** layer (or FiLM conditioning).
* **Outcome:** The Vision Encoder learns to dynamically shift its latent representations to "highlight" specific physical objects based on the semantic text prompt.

## 2. The Policy Head (Action Generation)
V1 is a *Forward World Model* (predicting future states). V2 requires an *Inverse Policy* (predicting motor actions).

* **Objective:** Map visual representations directly to motor control.
* **Architecture Upgrade:** 
    * **Freeze** the V1 ViT Encoder. Because V1 has already mastered mapping pixels to 3D physical coordinates, its weights do not need to be retrained.
    * Detach the V1 World Model.
    * Attach a new, lightweight Neural Network called the **Policy Head**.
* **Training:** 
    * Train the Policy Head using **Behavior Cloning (BC)** on human teleoperation datasets.
    * The Policy Head receives the ViT's Latent Vector + the Text Vector and predicts the necessary robotic actions to satisfy the text prompt.

## 3. Proprioception Injection
Visual data alone is insufficient for robust control; if the camera is temporarily occluded, the network must still know the physical position of its joints.

* **Objective:** Provide the network with internal awareness of its bodily configuration.
* **Architecture Upgrade:** 
    * Extract raw joint angles and velocities from the robot's hardware.
    * Feed this proprioceptive data directly into the Policy Head alongside the visual latent vector.
* **Outcome:** The policy can generate movements relative to its *current* known physical state, greatly increasing stability.

## 4. High-Frequency Delta Control
V1 predicts *absolute* 10D Cartesian space. Commanding a physical robot to move to absolute coordinates can cause sudden, dangerous jerks.

* **Objective:** Ensure buttery-smooth, safe physical movement.
* **Architecture Upgrade:** 
    * Switch the Policy Head output space from absolute coordinates to **Delta Actions** (e.g., `[dx, dy, dz, d_roll, d_pitch, d_yaw, d_grip]`).
* **Outcome:** The control loop can run at a high frequency (e.g., 20 Hz to 50 Hz), constantly streaming tiny, continuous adjustments to the servos.

## 5. Hardware Deployment (Jetson Orin Nano)
The robot must run entirely locally without relying on heavy desktop GPUs or cloud training loops.

* **Objective:** Optimize the network for real-time edge inference.
* **Architecture Upgrade:** 
    * Strip away all heavy JAX training loops, the JEPA World Model, and the Linear Probes.
    * Export the frozen ViT Encoder and the trained Policy Head using **TensorRT** (or ONNX), explicitly optimizing the FP16/INT8 kernels for the Jetson Orin Nano architecture.
* **Inference Pipeline:** 
    1. A lightweight Python/ROS node reads the USB Webcam and microphone.
    2. The TensorRT Engine processes the image and text to output delta-actions.
    3. The Jetson streams the delta-actions to the arm's motors in real-time.

---

### Summary of the V2 Execution Loop:
`User Prompt` $\rightarrow$ `Text Embedding`
`Webcam Image` $\rightarrow$ `Frozen ViT Encoder` $\rightarrow$ `Latent Physics Vector`
`(Text + Latent + Proprioception)` $\rightarrow$ `Policy Head` $\rightarrow$ `Delta Actions (20 Hz)` $\rightarrow$ `Robot Servos`
