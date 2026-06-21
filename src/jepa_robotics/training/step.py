import jax
import jax.numpy as jnp
import optax
from typing import Dict, Any, Tuple

def create_steps(encoder_def, predictor_def, world_model_def, probe_def, core_optimizer, probe_optimizer, loss_alpha: float = 1.0):
    """
    Returns JIT-compiled train and eval step functions bounded to the model definitions.
    """
    
    def loss_fn(encoder_params, predictor_params, wm_params, probe_params, target_params, batch: Dict[str, jnp.ndarray], rng: jax.Array):
        images = batch["image"]           # (B, S, H, W, C)
        actions = batch["action_7d"]      # (B, S, DoF)
        
        # 1. Perception Forward Pass
        b, s, h, w, c = images.shape
        flat_images = images.reshape((b * s, h, w, c))
        
        # Context Encoder (E_x)
        # We must provide a dropout rng for the random patch masking
        rng, dropout_rng = jax.random.split(rng)
        context_latents = encoder_def.apply(encoder_params, flat_images, train=True, rngs={'dropout': dropout_rng})
        context_latents = context_latents.reshape((b, s, -1))
        
        # Target Encoder (E_y) - Stop gradient, no dropout (train=False)
        target_latents = encoder_def.apply(target_params, flat_images, train=False)
        target_latents = jax.lax.stop_gradient(target_latents)
        target_latents = target_latents.reshape((b, s, -1))
        
        # Predictor (P) tries to guess the target latent of the current frame
        predicted_latents = predictor_def.apply(predictor_params, context_latents)
        latent_loss = jnp.mean((predicted_latents - target_latents) ** 2)
        
        # 2. World Model Forward Pass
        # We predict t+1 given t
        # Context inputs: t=0 to t=S-2
        seq_context = context_latents[:, :-1, :]
        seq_actions = actions[:, :-1, :]
        
        predicted_next_states = world_model_def.apply(wm_params, seq_context, seq_actions)
        
        # Temporal Dynamics Loss 
        # Compare against target latents at t=1 to t=S-1
        temporal_loss = jnp.mean((predicted_next_states - target_latents[:, 1:, :]) ** 2)
        
        # 3. Auxiliary Linear Probe (Interpretable Metric)
        sg_latents = jax.lax.stop_gradient(context_latents)
        predicted_states = probe_def.apply(probe_params, sg_latents)
        state_targets = batch["state_7d"]
        
        pos_mse = jnp.mean((predicted_states[..., :3] - state_targets[..., :3]) ** 2)
        rot_mse = jnp.mean((predicted_states[..., 3:6] - state_targets[..., 3:6]) ** 2)
        grip_mse = jnp.mean((predicted_states[..., 6:] - state_targets[..., 6:]) ** 2)
        
        # We still use mean MSE to optimize the probe's weights, as it's just a linear layer.
        probe_loss = jnp.mean((predicted_states - state_targets) ** 2)
        
        # 4. Total Loss for Optimizer
        total_loss = latent_loss + loss_alpha * temporal_loss + probe_loss
        
        metrics = {
            "loss": latent_loss + loss_alpha * temporal_loss,  # Only log the physics loss to the UI
            "latent_l2_loss": latent_loss,
            "temporal_dynamics_loss": temporal_loss,
            "probe_loss": probe_loss,
            "pos_mse": pos_mse,
            "rot_mse": rot_mse,
            "grip_mse": grip_mse
        }
        
        return total_loss, metrics

    @jax.jit
    def train_step(state: Dict[str, Any], batch: Dict[str, jnp.ndarray], tau: float) -> Tuple[Dict[str, Any], Dict[str, jnp.ndarray]]:
        """
        Executes a single forward/backward pass and updates optimizer states.
        """
        # Unpack state
        encoder_params = state["encoder_params"]
        predictor_params = state["predictor_params"]
        wm_params = state["wm_params"]
        probe_params = state["probe_params"]
        target_params = state["target_params"]
        core_opt_state = state["core_opt_state"]
        probe_opt_state = state["probe_opt_state"]
        rng = state["rng"]
        
        rng, step_rng = jax.random.split(rng)

        # Calculate gradients
        grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3), has_aux=True)
        (loss, metrics), grads = grad_fn(
            encoder_params, predictor_params, wm_params, probe_params, target_params, batch, step_rng
        )
        encoder_grads, predictor_grads, wm_grads, probe_grads = grads
        
        # 1. Update Core Network (ViT + Predictor + World Model)
        core_params_tuple = (encoder_params, predictor_params, wm_params)
        core_grads_tuple = (encoder_grads, predictor_grads, wm_grads)
        
        core_updates, new_core_opt_state = core_optimizer.update(core_grads_tuple, core_opt_state, core_params_tuple)
        new_core_params_tuple = optax.apply_updates(core_params_tuple, core_updates)
        new_encoder_params, new_predictor_params, new_wm_params = new_core_params_tuple
        
        # 2. Update Linear Probe Independently
        probe_updates, new_probe_opt_state = probe_optimizer.update(probe_grads, probe_opt_state, probe_params)
        new_probe_params = optax.apply_updates(probe_params, probe_updates)
        
        # Update Target Encoder via EMA
        from jepa_robotics.training.ema import update_target_ema
        new_target_params = update_target_ema(new_encoder_params, target_params, tau)
        
        # Pack state
        new_state = {
            "encoder_params": new_encoder_params,
            "predictor_params": new_predictor_params,
            "wm_params": new_wm_params,
            "probe_params": new_probe_params,
            "target_params": new_target_params,
            "core_opt_state": new_core_opt_state,
            "probe_opt_state": new_probe_opt_state,
            "rng": rng
        }
        
        return new_state, metrics

    @jax.jit
    def eval_step(state: Dict[str, Any], batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """
        Executes a pure forward pass for validation, computing loss metrics without any gradients or EMA updates.
        """
        encoder_params = state["encoder_params"]
        predictor_params = state["predictor_params"]
        wm_params = state["wm_params"]
        probe_params = state["probe_params"]
        target_params = state["target_params"]
        
        # We still need an RNG for the loss_fn signature, but since train=False we don't strictly need dropout
        rng = state["rng"]
        
        # Call loss_fn to get metrics
        total_loss, metrics = loss_fn(
            encoder_params, predictor_params, wm_params, probe_params, target_params, batch, rng
        )
        
        return metrics

    return train_step, eval_step
