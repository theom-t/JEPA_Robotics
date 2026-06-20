# JEPA Robotics - Coding & System Rules

These rules govern the development of the Cross-Embodiment Latent World Model and must be followed by all contributors, human and AI alike.

## 1. Mandatory Logging Rule
- **Logging Changes:** Any significant architectural changes, milestone completions, new feature additions, or major refactors MUST be logged in `log.md`.
- **Format:** Each entry must include the date and time, followed by a brief description of the change and the reasoning behind it. 

## 2. Hardware & Environment Restraints
- **Blackwell GPU Support:** Training targets the RTX 5090. Do NOT install `conda` CUDA packages as they conflict with the Blackwell architecture. Use your chosen ML framework's nightlies (e.g., `tf-nightly`) installed strictly via `pip`.
- **Edge Deployment Prep:** When building the V-JEPA and World Model, maintain modularity so that the perception encoder can be easily decoupled, quantized, and pruned for the Jetson Orin Nano later.

## 3. Project Guidelines
- **Project Structure:** Code should be organized into a clean `src/` or `jepa_robotics/` package layout. Separate data ingestion, model architecture, and training loops into distinct modules.
- **Type Hinting:** All Python functions must include clear type hints and return types.
- **Docstrings:** Use Google-style docstrings for all functions and classes.
- **Data Hygiene:** Always ensure data shapes and types are explicitly verified, especially when formatting the 7D Cartesian task space vectors.

## 4. Git & Version Control
- Commit messages must be descriptive and modular.
- Never commit large datasets (e.g., MP4s, Parquet files) or model checkpoints. Ensure `.gitignore` is properly configured.
