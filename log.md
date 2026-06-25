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

## 2026-06-23 19:31:00+01:00
- **Architectural Fix**: Upgraded to True V-JEPA Spatial Masking.
- **Details**:
  - **Problem Identified:** The previous masking implementation used `nn.Dropout` on patch embeddings, which zeroed individual neurons across ALL patches. This was not true masking — the Context Encoder (E_x) could still "see" every spatial position.
  - **Fix — `ViTEncoder`:** Removed `use_masking`/`masking_ratio` from the encoder. Added `patch_indices` parameter to `__call__`. Positional embeddings are now applied to ALL patches *before* the subset selection, preserving spatial position info. E_x is now genuinely blind to the target region (only `N_context` tokens processed). E_y continues to process all patches. Both paths now return `(patch_latents, pooled)` — patch-level for the JEPA loss, pooled for the World Model and Probe.
  - **Fix — `JEPAPredictor`:** Completely redesigned from a 1D MLP to a patch-level sequence predictor. Now accepts `(context_latents, target_pos_embeddings)`. A learnable mask token is injected at each target position with its spatial position embedding, then a small Transformer attends over the full `[context | target_tokens]` sequence. Only the target-position outputs are returned as predictions.
  - **Fix — `step.py`:** `loss_fn` now generates a true random spatial mask per step (`jax.random.permutation`), runs E_x on `context_indices` only, runs E_y on all patches (stop_gradient), extracts target positional embeddings from `target_params['params']['pos_embedding']`, and computes the JEPA L2 loss ONLY on the masked positions.
  - **Fix — `optimization.py`:** Removed the now-redundant `use_masking` Categorical and `EqualsCondition`. SMAC now tunes `masking_ratio` directly.
  - **Tests Updated:** `test_models.py` fully rewritten to validate the target path shape `(B, N_all, D)`, context path shape `(B, N_context, D)`, and the new patch-level predictor API.
- **Reasoning**: The original dropout-based approach was more akin to a noisy self-distillation (BYOL-style) than true V-JEPA. The spatial patch masking is the core mechanism that forces the model to learn genuine scene structure and physical reasoning from partial observations — which is critical for robust robotics perception and occlusion handling.

## 2026-06-23 19:47:00+01:00
- **Feature Addition**: Implemented SIGReg (Sketch Isotropic Gaussian Regularization).
- **Details**:
  - **Motivation**: The initial single training run (with the now-corrected true spatial masking) was stopped early due to the previously identified `nn.Dropout` masking issue. Additionally, the probe MSE (`pos_mse`) was observed to plateau early — a diagnostic signature of **dimensional collapse** in the latent space, where the network stops using most of the `latent_dim` dimensions. The EMA anti-collapse mechanism prevents the *target encoder* from dying but cannot prevent the *context encoder's pooled representation* from collapsing to a low-dimensional subspace.
  - **Theory (InfoMax)**: SIGReg completes the InfoMax objective. The JEPA reconstruction loss already minimises `H(Z|X)` (forces the latent to be informative about the input). SIGReg maximises `H(Z)` by pushing `p(z) → N(0, I)` — an isotropic Gaussian has maximum entropy for a fixed covariance, meaning every latent dimension is forced to carry independent, non-redundant information.
  - **Implementation**: A `sigreg_loss()` function was added to `step.py`. Per step, `num_sigreg_sketches=64` random unit vectors are sampled on the D-sphere. The pooled context latents `(B*S, D)` are projected onto each direction. Three moment-matching losses are applied: mean→0, variance→1, skewness→0. The "sketch" (random projection) approach is O(D × sketches) vs O(D²) for full covariance, making it JIT-efficient.
  - **Integration**: `sigreg_weight=0.1` added to the single-run config, loop, and SMAC search space `[0.01, 1.0]` (log-scale). The `sig_reg` metric is now logged per batch in the tqdm postfix.
  - **Tests**: Two unit tests added — one verifying loss ≈ 0 for true N(0,I) samples, one verifying loss is large (>10) for a completely collapsed (constant) representation.
- **Reasoning**: The combination of true spatial masking + SIGReg now gives us both the *structural* (patch-level spatial) and *statistical* (isotropic Gaussian) properties of a well-conditioned V-JEPA backbone. The system is now ready for the full single training run.

## 2026-06-23 19:56:00+01:00
- **Milestone**: SMAC3 Configuration Space Rebuilt for SOTA V1 Re-Sweep.
- **Details**: The previous SMAC sweep results are invalidated by the two architectural changes (true spatial masking + SIGReg). A full re-sweep is required. The following corrections were made to `optimization.py`:
  - **`patch_size=64` removed**: With 256×256 images this yields only 16 patches total. At `masking_ratio=0.75` E_x would see just 4 context patches — a degenerate configuration SMAC should never explore.
  - **`num_heads` extended to [4, 8, 16]**: 16 heads with `head_dim=32` is a strong SOTA choice for `latent_dim=512`. A `ForbiddenAndConjunction` clause prevents the invalid `(latent_dim=128, num_heads=16)` combination (`head_dim=8` is below the stable attention threshold).
  - **`weight_decay` range raised to `(1e-3, 0.1)`**: The old range `(1e-6, 1e-2)` capped SMAC below the SOTA ViT AdamW region (0.04–0.1 per MAE/V-JEPA/ViT papers).
  - **`learning_rate` narrowed to `(5e-5, 5e-4)`**: Centred on the V-JEPA effective range; the old upper bound of 1e-3 was wastefully exploring known-unstable territory.
  - **`batch_size` updated to [16, 32, 64]**: Removed 8 (too small for stable cross-embodiment gradients); added 64 (feasible on RTX 5090 with true masking's reduced peak token count).
  - **Hyperband budgets corrected**: `min=4, max=16, eta=2` gives clean power-of-2 rungs `4→8→16`. Old `max=10` with `eta=2` was a non-integer bracket.
  - **`n_trials` raised from 50 to 75**: Covers the expanded search space (new `sigreg_weight` dimension, changed dynamics).
  - **`cs.add_hyperparameters()` replaced with `cs.add()`**: Fixed SMAC deprecation warning.
  - **New run name `v1_true_masking_sigreg`**: Prevents collision with old `smac3_output/` results.
- **Reasoning**: The old incumbent (`masking_ratio=0.738, lr=5.976e-4, loss_alpha=6.457`) was found under a fundamentally different loss function (pooled L2, fake dropout masking, no SIGReg). These values cannot be safely reused for the true-masking patch-level system.


## 2026-06-24 22:46:00+01:00
- **Fix**: Normalised SMAC Probe Score Weighting in `loop.py`.
- **Details**: The old score (`pos * 1.0 + rot * 0.1 + grip * 0.1`) was effectively ignoring rotation — SMAC was optimising almost entirely for position accuracy. Analysis of V1 sweep output showed rot MSE (~0.21) is 35× larger than pos MSE (~0.006) in raw scale, so the 0.1× weight left rotation contributing near-zero to the objective. Replaced with a normalised score dividing each term by its observed typical range (pos/0.02, rot/0.25, grip/0.10) and weighting rotation equally to position (0.40 each, grip 0.20). This forces SMAC to discover architectures that genuinely learn rotation structure — physically critical for manipulation tasks where wrong orientation = task failure. The V1.5 carry-forward note in `docs/V2_plan.md` was removed as this is now implemented.
