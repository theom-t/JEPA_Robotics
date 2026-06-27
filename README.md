# JEPA Robotics: Cross-Embodiment Latent World Model

Welcome to the JEPA Robotics repository. This project aims to build an advanced, autonomous robotic system driven by a Self-Supervised Joint-Embedding Predictive Architecture (V-JEPA). 

The repository is structured into two phases: **V1** (The Foundation Visual Cortex) and **V2** (The Language-Conditioned Edge Robot).

---

## Hardware & Environment

This architecture pushes the absolute boundaries of consumer compute and edge deployment.

### 1. Training Rig (The Datacenter)
*   **GPU:** NVIDIA RTX 5090 (Blackwell Architecture).
*   **Constraint:** You **MUST NOT** install `conda` CUDA binaries, as they conflict with the Blackwell architecture.
*   **Requirement:** Install ML frameworks via `pip` strictly targeting nightlies (e.g., `jax` nightlies, `tf-nightly` CUDA 12.8/13.0 compat).

### 2. Edge Deployment (The Physical Robot)
*   **Computer:** NVIDIA Jetson Orin Nano (Configured for NVMe boot).
*   **Hardware Target:** A stationary table-mounted robotic arm (e.g., LeRobot SO100, WidowX 250).

---

## Setup Instructions

To get the environment running on your RTX 5090:

```bash
# 1. Create a Miniforge virtual environment
conda create -n jepa_robotics python=3.10
conda activate jepa_robotics

# 2. Install pip dependencies
# (Ensure you use the script that fetches the correct JAX/TF nightlies)
pip install -r requirements.txt

# 3. Download the Datasets
# Note: Ensure you have ample NVMe space (BridgeData V2 and LeRobot SO100)
python scripts/download_datasets.py
```

---

## Model & Design Choices

The core engine of this project is built on **JAX/Flax**, optimized for massive hardware utilization:

1.  **True V-JEPA Spatial Masking:** We use block masking (not noisy dropout) to drop 75% of the visual patches. The network *must* hallucinate the physical structure of the missing pieces to minimize latent L2 distance.
2.  **Anti-Collapse Mechanisms:** We prevent dimensional and spatial collapse using a strictly frozen EMA (Exponential Moving Average) Target Encoder.
3.  **Automatic Mixed Precision (`bfloat16`):** The entire forward pass is executed in `bfloat16` to leverage Blackwell Tensor Cores and halve VRAM usage, allowing us to double our batch sizes to 128. Gradients strictly remain in `float32` to prevent underflow.
4.  **Resumable Checkpointing:** The JAX PyTree (including `optax` AdamW momentums and PRNG keys) is fully serialized at the end of every epoch. You can hit `CTRL+C` and resume at any time.

---

## Roadmap: V1 (The Physics Engine)

Currently active. V1 is a self-supervised "passive observer" that watches robot datasets and learns the fundamental laws of physics.
*   **Goal:** Map 2D pixels to a highly structured 10D Cartesian latent space.
*   **World Model:** An Action-Conditioned Transformer that predicts future latent states based on simulated motor actions.
*   **Validation:** Uses an Interactive AI Debugger (running MJPEG EGL holograms) and a Headless Stress Tester to mathematically verify spatial stability and temporal momentum.

## Roadmap: V2 (The Autonomous Agent)

Planned implementation for Edge Robotics.
*   **Goal:** Convert the passive V1 engine into an active, language-driven controller.
*   **Language-Conditioned Behavior Cloning (LC-BC):** We will reuse the Open-X datasets (which already contain semantic language tags) to train a lightweight **Policy Head** with Cross-Attention.
*   **Edge Portability:** The massive training loops will be stripped away, and the frozen ViT Encoder + Policy Head will be compiled to a TensorRT engine to run at 30Hz+ locally on the Jetson Orin Nano.

---

*For detailed architectural specifics, review `architecture.md` and `docs/V2_plan.md`.*
