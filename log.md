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
