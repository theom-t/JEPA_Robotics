import jax
import jax.numpy as jnp
import optax
from typing import Dict, Any, Tuple


def sigreg_loss(
    latents: jnp.ndarray,
    rng: jax.Array,
    num_sketches: int = 64,
) -> jnp.ndarray:
    """
    Sketch Isotropic Gaussian Regularization (SIGReg).

    Maximizes information content by pushing the latent distribution towards
    an isotropic Gaussian N(0, I) via random 1D projections (sketching).

    Motivation:
        By the Cramér-Wold theorem, if ALL 1D projections of a distribution
        are Gaussian, the full distribution is Gaussian. Instead of computing
        the O(D²) full covariance, we test a random set of 1D projections —
        the "sketch" — making this O(D * num_sketches) and JIT-friendly.

        Enforcing N(0, I) maximises H(Z) (the marginal entropy of the latent
        space), completing the InfoMax objective alongside the JEPA loss which
        already minimises H(Z|X) via the reconstruction pressure.

    Args:
        latents: ``(N, D)`` pooled latent vectors.
        rng: JAX random key for sampling projection directions.
        num_sketches: Number of random 1D projections to evaluate.

    Returns:
        Scalar SIGReg loss. Zero when p(z) == N(0, I).
    """
    n, d = latents.shape

    # Sample num_sketches random unit vectors on the D-sphere
    raw_directions = jax.random.normal(rng, (d, num_sketches))
    norms = jnp.linalg.norm(raw_directions, axis=0, keepdims=True)
    directions = raw_directions / (norms + 1e-8)  # (D, num_sketches)

    # Project latents onto each direction → (N, num_sketches)
    projected = latents @ directions

    # 1st moment: push projected mean → 0
    proj_mean = jnp.mean(projected, axis=0)          # (num_sketches,)
    mean_loss = jnp.mean(proj_mean ** 2)

    # 2nd moment: push projected variance → 1
    proj_var = jnp.var(projected, axis=0)            # (num_sketches,)
    var_loss = jnp.mean((proj_var - 1.0) ** 2)

    # 3rd moment: push skewness → 0 (Gaussians are symmetric)
    proj_std = jnp.sqrt(proj_var + 1e-8)
    proj_centered = projected - proj_mean[None, :]   # (N, num_sketches)
    skewness = jnp.mean((proj_centered / proj_std[None, :]) ** 3, axis=0)
    skew_loss = jnp.mean(skewness ** 2)

    return mean_loss + var_loss + skew_loss


def spatial_sigreg_loss(latents_batch, rng, num_sketches=64):
    """
    Computes SIGReg independently across the spatial patches for EACH image in the batch.
    This strictly prevents 'Spatial Collapse', where the network maps every patch in an image
    to the exact same vector to cheat the JEPA loss.
    """
    b, n, d = latents_batch.shape
    raw_directions = jax.random.normal(rng, (d, num_sketches))
    norms = jnp.linalg.norm(raw_directions, axis=0, keepdims=True)
    directions = raw_directions / (norms + 1e-8)  # (D, num_sketches)
    
    # Project latents onto each direction → (B, N, num_sketches)
    projected = latents_batch @ directions
    
    # We want variance across N (spatial patches) for each image B
    proj_var = jnp.var(projected, axis=1)            # (B, num_sketches)
    var_loss = jnp.mean((proj_var - 1.0) ** 2)
    
    # We do NOT force the spatial mean to be 0 for each image individually, 
    # as images shouldn't be constrained to be zero-centered individually.
    # We only care about forcing spatial variance to 1.0!
    return var_loss
def create_steps(
    encoder_def,
    predictor_def,
    world_model_def,
    probe_def,
    core_optimizer,
    probe_optimizer,
    loss_alpha: float = 1.0,
    patch_size: int = 16,
    image_size: int = 256,
    masking_ratio: float = 0.75,
    sigreg_weight: float = 0.02,
    num_sigreg_sketches: int = 64,
    use_amp: bool = False,
):
    """
    Returns JIT-compiled train and eval step functions bound to the model definitions.

    Args:
        encoder_def: ViTEncoder module definition.
        predictor_def: JEPAPredictor module definition.
        world_model_def: ActionConditionedTransformer module definition.
        probe_def: StateLinearProbe module definition.
        core_optimizer: Optax optimizer for the core network (encoder + predictor + WM).
        probe_optimizer: Optax optimizer for the linear probe.
        loss_alpha: Weighting scalar for the temporal dynamics loss term.
        patch_size: Patch size used by the ViT (must match encoder_def.patch_size).
        image_size: Input image size in pixels (assumes square images).
        masking_ratio: Fraction of patches to mask as target (e.g. 0.75 masks 75%).
        sigreg_weight: Scalar weight for the SIGReg InfoMax regularization term.
        num_sigreg_sketches: Number of random 1D projections for the SIGReg sketch.
    """
    # --- Pre-compute static mask geometry (shapes must be fixed for JIT) ---
    num_patches: int = (image_size // patch_size) ** 2
    num_target: int = int(num_patches * masking_ratio)
    num_context: int = num_patches - num_target

    def loss_fn(
        encoder_params: Dict,
        predictor_params: Dict,
        wm_params: Dict,
        probe_params: Dict,
        target_params: Dict,
        batch: Dict[str, jnp.ndarray],
        rng: jax.Array,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Full V-JEPA + World Model forward pass and loss computation."""

        # --- Automatic Mixed Precision (AMP) System Switch ---
        if use_amp:
            images = jnp.nan_to_num(batch["image"]).astype(jnp.bfloat16)        # (B, S, H, W, C)
            enc_p = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), encoder_params)
            pred_p = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), predictor_params)
            wm_p = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), wm_params)
            targ_p = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), target_params)
            probe_p = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), probe_params)
        else:
            images = jnp.nan_to_num(batch["image"]).astype(jnp.float32)
            enc_p = encoder_params
            pred_p = predictor_params
            wm_p = wm_params
            targ_p = target_params
            probe_p = probe_params
            
        actions = jnp.nan_to_num(batch["action_10d"]).astype(jnp.float32)  # (B, S, 10)
        state_targets = jnp.nan_to_num(batch["state_10d"]).astype(jnp.float32)

        b, s, h, w, c = images.shape
        flat_images = images.reshape((b * s, h, w, c))  # (B*S, H, W, C)

        # ── 1. Generate a Contiguous Spatial Mask (Block Masking) ────────────
        # Instead of random speckling (which allows algebraic interpolation), we drop a 
        # contiguous blob of patches. This forces the model to actually hallucinate structures.
        rng, mask_rng = jax.random.split(rng)
        rng, sigreg_rng = jax.random.split(rng)
        
        grid_size = image_size // patch_size
        x = jnp.arange(grid_size)
        y = jnp.arange(grid_size)
        xx, yy = jnp.meshgrid(x, y)
        
        # Pick a random center for the masked block
        cx = jax.random.uniform(mask_rng, minval=0.0, maxval=grid_size)
        cy = jax.random.uniform(mask_rng, minval=0.0, maxval=grid_size)
        
        # Calculate squared distance from center to form a circle/blob
        distances = (xx - cx)**2 + (yy - cy)**2
        flat_distances = distances.flatten()
        
        # Add tiny uniform noise to break discrete distance ties randomly
        noise = jax.random.uniform(mask_rng, shape=flat_distances.shape, minval=0.0, maxval=0.1)
        flat_distances = flat_distances + noise
        
        # Sort by distance: patches closest to the center become the TARGET (masked)
        sorted_indices = jnp.argsort(flat_distances)
        
        target_indices = sorted_indices[:num_target]   # (N_target,)  — masked patches
        context_indices = sorted_indices[num_target:]  # (N_context,) — visible patches

        # ── 2. Context Encoder (E_x) ─────────────────────────────────────────
        # E_x is BLIND to the masked region — it only processes context patches.
        # This forces the Predictor to genuinely infer missing spatial content.
        context_patch_latents, context_pooled = encoder_def.apply(
            enc_p,
            flat_images,
            patch_indices=context_indices,
        )
        # context_patch_latents: (B*S, N_context, D)
        # context_pooled:        (B*S, D)

        # ── 3. Target Encoder (E_y) ───────────────────────────────────────────
        # E_y sees ALL patches — it is the EMA teacher, providing clean targets.
        # stop_gradient: E_y is never updated by backprop, only via EMA.
        target_patch_latents, target_pooled = encoder_def.apply(
            targ_p,
            flat_images,
            patch_indices=None,  # All patches — no masking on the target path
        )
        target_patch_latents = jax.lax.stop_gradient(target_patch_latents)
        target_pooled = jax.lax.stop_gradient(target_pooled)
        # target_patch_latents: (B*S, N_patches, D)

        # Select only the masked (target) patch latents — these are what we predict
        target_patch_latents_masked = target_patch_latents[:, target_indices, :]
        # (B*S, N_target, D)

        # ── 4. Extract Target Positional Embeddings ───────────────────────────
        # The Predictor needs to know WHERE in the image the masked patches are.
        # We reuse the encoder's learned positional embedding table (from target_params
        # so the positions match the E_y feature space). Sliced to target positions.
        pos_embedding = targ_p["params"]["pos_embedding"]  # (1, N_patches, D)
        target_pos_emb = pos_embedding[:, target_indices, :]       # (1, N_target, D)
        target_pos_emb = jnp.broadcast_to(
            target_pos_emb, (b * s, num_target, encoder_def.latent_dim)
        )  # (B*S, N_target, D)

        # ── 5. Predictor (P) ─────────────────────────────────────────────────
        # P receives visible context latents + positional hints for the masked region,
        # and predicts what E_y would have produced at those masked positions.
        predicted_target_latents = predictor_def.apply(
            pred_p,
            context_patch_latents,
            target_pos_emb,
        )  # (B*S, N_target, D)
        
        # CAST BACK TO FLOAT32 BEFORE LOSS COMPUTATION
        # Computing loss gradients in bfloat16 causes catastrophic quantization and collapse!
        predicted_target_latents = predicted_target_latents.astype(jnp.float32)
        target_patch_latents_masked = target_patch_latents_masked.astype(jnp.float32)

        # ── 6. JEPA Latent Loss ───────────────────────────────────────────────
        # L2-normalize latents to force them onto a unit hypersphere.
        # This makes representation collapse to 0 mathematically impossible.
        pred_norm = predicted_target_latents / (jnp.linalg.norm(predicted_target_latents, axis=-1, keepdims=True) + 1e-8)
        targ_norm = target_patch_latents_masked / (jnp.linalg.norm(target_patch_latents_masked, axis=-1, keepdims=True) + 1e-8)
        latent_loss = jnp.mean((pred_norm - targ_norm) ** 2)

        # ── 7. World Model Forward Pass ───────────────────────────────────────
        # The World Model reasons over pooled temporal latent sequences.
        # We use CONTEXT pooled latents (what the model actually sees per frame)
        # and predict the TARGET pooled latent of the next frame (ground truth physics).
        context_pooled_seq = context_pooled.reshape((b, s, -1))  # (B, S, D)
        target_pooled_seq = target_pooled.reshape((b, s, -1))    # (B, S, D)

        seq_context = context_pooled_seq[:, :-1, :]  # (B, S-1, D) — frames t=0..S-2
        seq_actions = actions[:, :-1, :]             # (B, S-1, 10)

        predicted_next_states = world_model_def.apply(wm_p, seq_context, seq_actions)
        # (B, S-1, D)
        
        predicted_next_states = predicted_next_states.astype(jnp.float32)
        target_pooled_seq_f32 = target_pooled_seq.astype(jnp.float32)

        # Temporal Dynamics Loss: compare predicted next state vs. true E_y next state
        # L2-normalize to prevent World Model from inducing collapse
        wm_pred_norm = predicted_next_states / (jnp.linalg.norm(predicted_next_states, axis=-1, keepdims=True) + 1e-8)
        wm_targ_norm = target_pooled_seq_f32[:, 1:, :] / (jnp.linalg.norm(target_pooled_seq_f32[:, 1:, :], axis=-1, keepdims=True) + 1e-8)
        
        temporal_loss = jnp.mean((wm_pred_norm - wm_targ_norm) ** 2)

        # ── 8. Auxiliary Linear Probe ─────────────────────────────────────────
        # Regresses the 10D physical state from the pooled context latents.
        # Operates on a stop_gradient copy — the probe never affects the encoder.
        sg_latents = jax.lax.stop_gradient(context_pooled_seq)  # (B, S, D)
        predicted_states = probe_def.apply(probe_p, sg_latents)  # (B, S, 10)
        
        # Cast predictions back to float32 before computing MSE to preserve numerical stability
        predicted_states = predicted_states.astype(jnp.float32)
        
        pos_diff = predicted_states[..., :3] - state_targets[..., :3]
        rot_diff = predicted_states[..., 3:9] - state_targets[..., 3:9]
        grip_diff = predicted_states[..., 9:] - state_targets[..., 9:]

        # 6D rotation uses continuous matrices — MSE correctly measures alignment
        pos_mse = jnp.mean(pos_diff ** 2)
        rot_mse = jnp.mean(rot_diff ** 2)
        grip_mse = jnp.mean(grip_diff ** 2)

        wrapped_diffs = jnp.concatenate([pos_diff, rot_diff, grip_diff], axis=-1)
        probe_loss = jnp.mean(wrapped_diffs ** 2)

        # ── 9. SIGReg (Sketch Isotropic Gaussian Regularization) ─────────────
        # BATCH Variance: Prevents different images from collapsing to the same global state
        flat_pooled = context_pooled_seq.reshape((b * s, -1))  # (B*S, D)
        rng, sigreg_rng_batch = jax.random.split(rng)
        sig_reg_batch = sigreg_loss(flat_pooled, sigreg_rng_batch, num_sketches=num_sigreg_sketches)
        
        # SPATIAL Variance: Prevents all patches within an image from collapsing to the same local state
        # context_patch_latents: (B*S, N_context, D)
        rng, sigreg_rng_spatial = jax.random.split(rng)
        sig_reg_spatial = spatial_sigreg_loss(context_patch_latents, sigreg_rng_spatial, num_sketches=num_sigreg_sketches)
        
        sig_reg = sig_reg_batch + sig_reg_spatial

        # ── 10. Total Loss (Computed in Float32) ──────────────────────────────
        total_loss = latent_loss + loss_alpha * temporal_loss + probe_loss + sigreg_weight * sig_reg

        metrics = {
            "loss": latent_loss + loss_alpha * temporal_loss,  # Core physics signal
            "latent_l2_loss": latent_loss,
            "temporal_dynamics_loss": temporal_loss,
            "probe_loss": probe_loss,
            "sig_reg": sig_reg,
            "pos_mse": pos_mse,
            "rot_mse": rot_mse,
            "grip_mse": grip_mse,
        }

        return total_loss, metrics

    @jax.jit
    def train_step(
        state: Dict[str, Any],
        batch: Dict[str, jnp.ndarray],
        tau: float,
    ) -> Tuple[Dict[str, Any], Dict[str, jnp.ndarray]]:
        """
        Executes a single forward/backward pass, updates all optimizer states,
        and applies the EMA update to the Target Encoder.
        """
        encoder_params = state["encoder_params"]
        predictor_params = state["predictor_params"]
        wm_params = state["wm_params"]
        probe_params = state["probe_params"]
        target_params = state["target_params"]
        core_opt_state = state["core_opt_state"]
        probe_opt_state = state["probe_opt_state"]
        rng = state["rng"]

        rng, step_rng = jax.random.split(rng)

        # Differentiate w.r.t. encoder, predictor, WM, and probe params
        grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3), has_aux=True)
        (loss, metrics), grads = grad_fn(
            encoder_params, predictor_params, wm_params, probe_params,
            target_params, batch, step_rng,
        )
        encoder_grads, predictor_grads, wm_grads, probe_grads = grads

        # Update Core Network (Context Encoder + Predictor + World Model)
        core_params_tuple = (encoder_params, predictor_params, wm_params)
        core_grads_tuple = (encoder_grads, predictor_grads, wm_grads)

        core_updates, new_core_opt_state = core_optimizer.update(
            core_grads_tuple, core_opt_state, core_params_tuple
        )
        new_core_params_tuple = optax.apply_updates(core_params_tuple, core_updates)
        new_encoder_params, new_predictor_params, new_wm_params = new_core_params_tuple

        # Update Linear Probe Independently (separate learning rate)
        probe_updates, new_probe_opt_state = probe_optimizer.update(
            probe_grads, probe_opt_state, probe_params
        )
        new_probe_params = optax.apply_updates(probe_params, probe_updates)

        # Target Encoder (E_y) updated via EMA — never via backprop
        from jepa_robotics.training.ema import update_target_ema
        new_target_params = update_target_ema(new_encoder_params, target_params, tau)

        new_state = {
            "encoder_params": new_encoder_params,
            "predictor_params": new_predictor_params,
            "wm_params": new_wm_params,
            "probe_params": new_probe_params,
            "target_params": new_target_params,
            "core_opt_state": new_core_opt_state,
            "probe_opt_state": new_probe_opt_state,
            "rng": rng,
        }

        return new_state, metrics

    @jax.jit
    def eval_step(
        state: Dict[str, Any],
        batch: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """
        Pure forward pass for validation — no gradients or EMA updates.
        Uses the same masking logic for a fair and consistent evaluation.
        """
        encoder_params = state["encoder_params"]
        predictor_params = state["predictor_params"]
        wm_params = state["wm_params"]
        probe_params = state["probe_params"]
        target_params = state["target_params"]
        rng = state["rng"]

        _, metrics = loss_fn(
            encoder_params, predictor_params, wm_params, probe_params,
            target_params, batch, rng,
        )
        return metrics

    return train_step, eval_step
