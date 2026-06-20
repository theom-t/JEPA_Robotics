# JEPA Robotics Project Rules

- **Logging**: You MUST log any large changes, milestone completions, or major refactors in `log.md` with the date, time, and a description.
- **Code Style**: Use type hinting, a clean modular layout, and Google-style docstrings.
- **Environment**: Do NOT install conda CUDA packages; use pip for your chosen ML framework nightlies (e.g., `tf-nightly`) to maintain compatibility with the user's RTX 5090 (Blackwell) setup.
