import jax
import jax.numpy as jnp
import optax
from typing import Dict, Any, Tuple

def create_train_step(encoder_def, predictor_def, world_model_def, optimizer):
    """
    Returns a JIT-compiled training step function bounded to the model definitions.
    """
    
    def loss_fn(encoder_params, predictor_params, wm_params, target_params, batch: Dict[str, jnp.ndarray]):
        images = batch["image"]           # (B, S, H, W, C)
        actions = batch["action_7d"]      # (B, S, DoF)
        
        # 1. Perception Forward Pass
        b, s, h, w, c = images.shape
        flat_images = images.reshape((b * s, h, w, c))
        
        # Context Encoder (E_x)
        context_latents = encoder_def.apply(encoder_params, flat_images)
        context_latents = context_latents.reshape((b, s, -1))
        
        # Target Encoder (E_y) - Stop gradient
        target_latents = encoder_def.apply(target_params, flat_images)
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
        
        # 3. Total Loss
        total_loss = latent_loss + temporal_loss
        
        metrics = {
            "loss": total_loss,
            "latent_l2_loss": latent_loss,
            "temporal_dynamics_loss": temporal_loss
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
        target_params = state["target_params"]
        opt_state = state["opt_state"]

        # Calculate gradients
        grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True)
        (loss, metrics), grads = grad_fn(
            encoder_params, predictor_params, wm_params, target_params, batch
        )
        encoder_grads, predictor_grads, wm_grads = grads
        
        # Combine parameters and gradients into a flat structure for optax
        params_tuple = (encoder_params, predictor_params, wm_params)
        grads_tuple = (encoder_grads, predictor_grads, wm_grads)
        
        # Apply gradients
        updates, new_opt_state = optimizer.update(grads_tuple, opt_state, params_tuple)
        new_params_tuple = optax.apply_updates(params_tuple, updates)
        
        new_encoder_params, new_predictor_params, new_wm_params = new_params_tuple
        
        # Update Target Encoder via EMA
        from jepa_robotics.training.ema import update_target_ema
        new_target_params = update_target_ema(new_encoder_params, target_params, tau)
        
        # Pack state
        new_state = {
            "encoder_params": new_encoder_params,
            "predictor_params": new_predictor_params,
            "wm_params": new_wm_params,
            "target_params": new_target_params,
            "opt_state": new_opt_state
        }
        
        return new_state, metrics

    return train_step
