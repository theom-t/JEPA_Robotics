# Project Log: JEPA Robotics

*All significant architectural changes, milestone completions, and major refactors must be logged here with a date, time, and description.*

---

## 2026-06-20 15:45:00+01:00
- **Initialization**: Set up core project structure.
- **Details**: Created `GEMINI.md` for project rules, `.gitignore` for standard Python/ML exclusions, and `environment.yml`. 
- **Reasoning**: To establish clean project hygiene and ensure the AI/Human team adheres to RTX 5090 environment constraints and structural best practices from day one.

## 2026-06-20 15:47:00+01:00
- **Framework Agnostic Update**: Removed PyTorch assumptions.
- **Details**: Removed PyTorch-specific nightlies from `environment.yml` and `.gitignore`. Updated `GEMINI.md` and `.agents/AGENTS.md` to be framework-agnostic.
- **Reasoning**: To prevent prematurely locking the project into PyTorch before the final framework decision (e.g., JAX or tf-nightly) is made.

## 2026-06-20 16:10:00+01:00
- **Feature Addition**: Implemented OOP Data Ingestion & Kinematic Pipeline.
- **Details**: Created `src/jepa_robotics/data/` module containing `kinematics.py` (for 7D Cartesian bridging) and `dataset_loaders.py` (OOP streaming for BridgeData V2 and SO100). Added `scripts/download_datasets.py` with data reduction logic (`--limit`) and `scripts/visualize_batch.py` for visual testing.
- **Reasoning**: To establish a robust, efficient data loading architecture that standardizes inputs before we begin modeling the JEPA latent space.

## 2026-06-22 20:45:00+01:00
- **Milestone**: V1 Architecture Stabilization, SMAC Evaluation, & V2 Roadmap Generation.
- **Details**: 
  - Upgraded the State Predictor to use continuous **6D Rotation Matrices** instead of Euler angles, directly preventing rotational wrap-around anomalies.
  - Validated JAX architecture against temporal information leakage (verified causal masking in WM and frame-independence in ViT).
  - Executed extensive SMAC3 optimization sweep, locking best Weighted Probe Score at ~0.037.
  - Scripted `smac_assessment.py` to auto-generate convergence and hyperparameter trajectory graphs.
  - Hardened SMAC objective: Prevented the optimizer from "cheating" by enforcing `use_masking=True` and bounding `masking_ratio` to `[0.5, 0.95]`.
  - Added `orbax.checkpoint` support to serialize and save the final models locally in `single` training mode for edge inference.
  - Added Virtual Sandbox Evaluation stage to V1 planning docs.
  - Created initial `V2_plan.md` architectural blueprint (Language-Conditioned Policy Head running on edge devices).
- **Reasoning**: We have verified that the V-JEPA latent world model is properly predicting physics without hallucinating or leaking. By bounding the masking ratio, we ensure the ViT learns powerful semantics rather than acting as a lazy image processor. We are now fully prepared to train the final V1 backbone on 100 epochs, save the weights via Orbax, and evaluate it in the upcoming Mujoco Virtual Sandbox before building the V2 Policy Head.
