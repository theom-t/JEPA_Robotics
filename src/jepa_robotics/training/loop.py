import jax
import jax.numpy as jnp
import optax
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
import os
import orbax.checkpoint as ocp

from jepa_robotics.models.v_jepa import ViTEncoder, JEPAPredictor, StateLinearProbe
from jepa_robotics.models.world_model import ActionConditionedTransformer
from jepa_robotics.training.step import create_steps
from jepa_robotics.data.dataset_loaders import BridgeDataLoader, SO100DataLoader

def train_model(config: Dict[str, Any], num_epochs: int = 1, do_eval: bool = True, save_dir: str = None) -> float:
    """
    Core orchestrator that binds dataloaders, compiles the network, 
    and executes the training loop.
    Returns the final validation loss.
    """
    # 1. Telemetry is now disabled to run locally without a cloud account
    
    # 2. Instantiate Dynamic Architectures
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    latent_dim = config["latent_dim"]
    vit_depth = config["vit_depth"]
    wm_depth = config["wm_depth"]
    num_heads = config["num_heads"]
    lr = config["learning_rate"]
    tau = config["tau"]
    
    # New extracted hyperparameters
    patch_size = config.get("patch_size", 16)
    use_masking = config.get("use_masking", True)
    masking_ratio = config.get("masking_ratio", 0.7) if use_masking else 0.0
    activation_fn = config.get("activation_fn", "gelu")
    weight_decay = config.get("weight_decay", 1e-4)
    probe_lr = config.get("probe_learning_rate", 1e-3)
    batch_size = config.get("batch_size", 32)
    seq_len = config.get("seq_len", 5)
    loss_alpha = config.get("loss_alpha", 1.0)
    
    encoder_def = ViTEncoder(
        latent_dim=latent_dim, depth=vit_depth, num_heads=num_heads, 
        patch_size=patch_size, activation_fn=activation_fn, 
        use_masking=use_masking, masking_ratio=masking_ratio
    )
    predictor_def = JEPAPredictor(
        latent_dim=latent_dim, activation_fn=activation_fn
    )
    wm_def = ActionConditionedTransformer(
        latent_dim=latent_dim, depth=wm_depth, num_heads=num_heads, 
        activation_fn=activation_fn
    )
    probe_def = StateLinearProbe(out_dim=10)
    
    # Mock inputs for initialization (Batch=batch_size, SeqLen-1 for World Model)
    mock_img = jnp.ones((1, 256, 256, 3))
    mock_seq_latents = jnp.ones((batch_size, seq_len - 1, latent_dim))
    mock_seq_actions = jnp.ones((batch_size, seq_len - 1, 10))
    
    init_rngs = {'params': init_rng, 'dropout': init_rng}
    
    # Initialize parameters
    encoder_params = encoder_def.init(init_rngs, mock_img, train=False)
    target_params = encoder_def.init(init_rngs, mock_img, train=False) # Clone structure
    predictor_params = predictor_def.init(init_rng, jnp.ones((1, latent_dim)))
    wm_params = wm_def.init(init_rng, mock_seq_latents, mock_seq_actions)
    probe_params = probe_def.init(init_rng, jnp.ones((1, latent_dim)))
    
    # 3. Setup Optax Optimizers (Decoupled & NaN-Protected)
    core_optimizer = optax.chain(
        optax.clip(10.0), # Safe scalar clipping. Global norm squaring can cause float32 'inf' overflows
        optax.zero_nans(),
        optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    )
    probe_optimizer = optax.chain(
        optax.clip(10.0),
        optax.zero_nans(),
        optax.adamw(learning_rate=probe_lr, weight_decay=0.0)
    )
    
    core_params_tuple = (encoder_params, predictor_params, wm_params)
    core_opt_state = core_optimizer.init(core_params_tuple)
    probe_opt_state = probe_optimizer.init(probe_params)
    
    # Pack into state dictionary
    state = {
        "encoder_params": encoder_params,
        "predictor_params": predictor_params,
        "wm_params": wm_params,
        "probe_params": probe_params,
        "target_params": target_params,
        "core_opt_state": core_opt_state,
        "probe_opt_state": probe_opt_state,
        "rng": rng
    }
    
    # 4. Compile JAX Train and Eval Steps
    train_step_fn, eval_step_fn = create_steps(encoder_def, predictor_def, wm_def, probe_def, core_optimizer, probe_optimizer, loss_alpha)
    
    # 5. Initialize Real Dataloaders (Sliding Window & Batching)
    is_smac = config.get("is_smac_run", False)
    # Allow overriding sample_fraction, default to 4% for SMAC or 100% otherwise
    sample_fraction = config.get("sample_fraction", 0.04 if is_smac else 1.0)
    
    # We also enforce a hard ceiling on batches per epoch during SMAC so it never hangs
    max_train_batches = 1000 if is_smac else None
    max_val_batches = 200 if is_smac else None
    
    bridge_loader = BridgeDataLoader(batch_size=batch_size, seq_len=seq_len, sample_fraction=sample_fraction)
    so100_loader = SO100DataLoader(batch_size=batch_size, seq_len=seq_len, sample_fraction=sample_fraction)
    
    if do_eval:
        bridge_val_loader = BridgeDataLoader(batch_size=batch_size, seq_len=seq_len, sample_fraction=sample_fraction)
        so100_val_loader = SO100DataLoader(batch_size=batch_size, seq_len=seq_len, sample_fraction=sample_fraction)
    
    final_loss = 0.0
    
    # 6. Execute Alternating Training Loop
    print(f"\\nStarting Training Run (Latent: {latent_dim}, Epochs: {num_epochs}, Patch: {patch_size}, Masking: {use_masking})...")
    
    def cycle_loader(loader, split):
        """Yields batches continuously by restarting the loader when it hits StopIteration."""
        while True:
            for batch in loader.load(split=split):
                yield batch

    for epoch in range(num_epochs):
        bridge_iter = bridge_loader.load(split="train")
        so100_iter = cycle_loader(so100_loader, split="train")
        
        epoch_losses = []
        pbar = tqdm(zip(bridge_iter, so100_iter), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", total=max_train_batches)
        
        for batch_idx, (bridge_batch, so100_batch) in enumerate(pbar):
            if max_train_batches and batch_idx >= max_train_batches:
                break
            # Process BridgeData
            state, metrics_b = train_step_fn(state, bridge_batch, tau)
            
            # Process SO100 (Cross-Embodiment sharing the SAME state/weights!)
            state, metrics_s = train_step_fn(state, so100_batch, tau)
            
            # Log telemetry
            avg_loss = (metrics_b["loss"] + metrics_s["loss"]) / 2.0
            epoch_losses.append(avg_loss)
            
            # Log telemetry locally via tqdm instead of spamming print statements
            avg_pos_mse = (metrics_b["pos_mse"] + metrics_s["pos_mse"]) / 2.0
            avg_rot_mse = (metrics_b["rot_mse"] + metrics_s["rot_mse"]) / 2.0
            avg_grip_mse = (metrics_b["grip_mse"] + metrics_s["grip_mse"]) / 2.0
            
            pbar.set_postfix({
                "Avg L": f"{avg_loss:.3f}",
                "Pos": f"{avg_pos_mse:.3f}",
                "Rot": f"{avg_rot_mse:.3f}",
                "Grp": f"{avg_grip_mse:.3f}"
            })
            
        final_loss = np.mean(epoch_losses)
        print(f"\\n✅ Epoch {epoch+1} Train Completed - Avg Loss: {final_loss:.4f}\\n")
        
        if do_eval:
            val_epoch_losses = []
            val_pos_mses = []
            val_rot_mses = []
            val_grip_mses = []
            
            bridge_val_iter = bridge_val_loader.load(split="val")
            so100_val_iter = cycle_loader(so100_val_loader, split="val")
            
            pbar_val = tqdm(zip(bridge_val_iter, so100_val_iter), desc=f"Epoch {epoch+1} Validation", unit="batch", total=max_val_batches)
            for batch_idx_val, (bridge_val_batch, so100_val_batch) in enumerate(pbar_val):
                if max_val_batches and batch_idx_val >= max_val_batches:
                    break
                metrics_b_val = eval_step_fn(state, bridge_val_batch)
                metrics_s_val = eval_step_fn(state, so100_val_batch)
                
                avg_val_loss = (metrics_b_val["loss"] + metrics_s_val["loss"]) / 2.0
                avg_val_pos = (metrics_b_val["pos_mse"] + metrics_s_val["pos_mse"]) / 2.0
                avg_val_rot = (metrics_b_val["rot_mse"] + metrics_s_val["rot_mse"]) / 2.0
                avg_val_grip = (metrics_b_val["grip_mse"] + metrics_s_val["grip_mse"]) / 2.0
                
                val_epoch_losses.append(avg_val_loss)
                val_pos_mses.append(avg_val_pos)
                val_rot_mses.append(avg_val_rot)
                val_grip_mses.append(avg_val_grip)
                
                pbar_val.set_postfix({
                    "Val L": f"{avg_val_loss:.3f}",
                    "Pos": f"{avg_val_pos:.3f}",
                    "Rot": f"{avg_val_rot:.3f}",
                    "Grp": f"{avg_val_grip:.3f}"
                })
                
            final_val_loss = np.mean(val_epoch_losses)
            final_val_pos = np.mean(val_pos_mses)
            final_val_rot = np.mean(val_rot_mses)
            final_val_grip = np.mean(val_grip_mses)
            
            # Create a weighted score so large rotation radian errors don't crush small positional meter errors
            weighted_probe_score = final_val_pos * 1.0 + final_val_rot * 0.1 + final_val_grip * 0.1
            
            print(f"\\n🎯 Epoch {epoch+1} Validation Completed - Val Avg Loss: {final_val_loss:.4f} | Pos: {final_val_pos:.4f} | Rot: {final_val_rot:.4f} | Grp: {final_val_grip:.4f} | SMAC Score: {weighted_probe_score:.4f}\\n")
            
            # If evaluating, return the weighted PROBE MSE to SMAC.
            final_loss = weighted_probe_score
            
            # Early stopping: if the loss diverges to NaN, instantly abort the trial
            # and return a massive penalty so SMAC learns not to use this learning rate
            if np.isnan(final_loss):
                print("\n[WARNING] Loss diverged to NaN. Aborting trial early to save compute.\n")
                return 999.0
            
    if save_dir is not None:
        print(f"\n[INFO] Saving final model checkpoint to {save_dir}...")
        os.makedirs(save_dir, exist_ok=True)
        
        save_state = {
            "encoder_params": state["encoder_params"],
            "predictor_params": state["predictor_params"],
            "wm_params": state["wm_params"],
            "probe_params": state["probe_params"]
        }
        
        # Orbax can crash on JAX nightlies due to internal API changes. 
        # Using standard flax serialization instead.
        import flax.serialization
        save_path = os.path.join(save_dir, "v1_weights.msgpack")
        with open(save_path, "wb") as f:
            f.write(flax.serialization.to_bytes(save_state))
            
        print(f"[INFO] Model successfully saved to {save_path}!\n")
            
    # Return loss for SMAC3 Pareto evaluation
    return float(final_loss)
