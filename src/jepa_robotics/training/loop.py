import jax
import jax.numpy as jnp
import optax
import numpy as np
from tqdm import tqdm
from typing import Dict, Any

from jepa_robotics.models.v_jepa import ViTEncoder, JEPAPredictor, StateLinearProbe
from jepa_robotics.models.world_model import ActionConditionedTransformer
from jepa_robotics.training.step import create_steps
from jepa_robotics.data.dataset_loaders import BridgeDataLoader, SO100DataLoader

def train_model(config: Dict[str, Any], num_epochs: int = 1, do_eval: bool = True) -> float:
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
    probe_def = StateLinearProbe(out_dim=7)
    
    # Mock inputs for initialization (Batch=batch_size, SeqLen-1 for World Model)
    mock_img = jnp.ones((1, 256, 256, 3))
    mock_seq_latents = jnp.ones((batch_size, seq_len - 1, latent_dim))
    mock_seq_actions = jnp.ones((batch_size, seq_len - 1, 7))
    
    init_rngs = {'params': init_rng, 'dropout': init_rng}
    
    # Initialize parameters
    encoder_params = encoder_def.init(init_rngs, mock_img, train=False)
    target_params = encoder_def.init(init_rngs, mock_img, train=False) # Clone structure
    predictor_params = predictor_def.init(init_rng, jnp.ones((1, latent_dim)))
    wm_params = wm_def.init(init_rng, mock_seq_latents, mock_seq_actions)
    probe_params = probe_def.init(init_rng, jnp.ones((1, latent_dim)))
    
    # 3. Setup Optax Optimizer
    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    
    params_tuple = (encoder_params, predictor_params, wm_params, probe_params)
    opt_state = optimizer.init(params_tuple)
    
    # Pack into state dictionary
    state = {
        "encoder_params": encoder_params,
        "predictor_params": predictor_params,
        "wm_params": wm_params,
        "probe_params": probe_params,
        "target_params": target_params,
        "opt_state": opt_state,
        "rng": rng
    }
    
    # 4. Compile JAX Train and Eval Steps
    train_step_fn, eval_step_fn = create_steps(encoder_def, predictor_def, wm_def, probe_def, optimizer, loss_alpha)
    
    # 5. Initialize Real Dataloaders (Sliding Window & Batching)
    # If SMAC is running, we stratify a 10% slice to ensure sweeps complete quickly.
    # Otherwise, we use 100% of the dataset for the final multi-day training.
    sample_fraction = 0.10 if config.get("is_smac_run", False) else 1.0
    
    bridge_loader = BridgeDataLoader(batch_size=batch_size, seq_len=seq_len, sample_fraction=sample_fraction)
    so100_loader = SO100DataLoader(batch_size=batch_size, seq_len=seq_len, sample_fraction=sample_fraction)
    
    if do_eval:
        bridge_val_loader = BridgeDataLoader(batch_size=batch_size, seq_len=seq_len, sample_fraction=sample_fraction)
        so100_val_loader = SO100DataLoader(batch_size=batch_size, seq_len=seq_len, sample_fraction=sample_fraction)
    
    final_loss = 0.0
    
    # 6. Execute Alternating Training Loop
    print(f"\\nStarting Training Run (Latent: {latent_dim}, Epochs: {num_epochs}, Patch: {patch_size}, Masking: {use_masking})...")
    for epoch in range(num_epochs):
        bridge_iter = bridge_loader.load()
        so100_iter = so100_loader.load()
        
        # We zip them to alternate batches cleanly
        epoch_losses = []
        pbar = tqdm(zip(bridge_iter, so100_iter), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for bridge_batch, so100_batch in pbar:
            # Process BridgeData
            state, metrics_b = train_step_fn(state, bridge_batch, tau)
            
            # Process SO100 (Cross-Embodiment sharing the SAME state/weights!)
            state, metrics_s = train_step_fn(state, so100_batch, tau)
            
            # Log telemetry
            avg_loss = (metrics_b["loss"] + metrics_s["loss"]) / 2.0
            epoch_losses.append(avg_loss)
            
            # Log telemetry locally via tqdm instead of spamming print statements
            avg_probe_mse = (metrics_b["probe_loss"] + metrics_s["probe_loss"]) / 2.0
            pbar.set_postfix({
                "Bridge L": f"{metrics_b['loss']:.3f}", 
                "SO100 L": f"{metrics_s['loss']:.3f}", 
                "Avg L": f"{avg_loss:.3f}",
                "Probe MSE": f"{avg_probe_mse:.3f}"
            })
            
        final_loss = np.mean(epoch_losses)
        print(f"\\n✅ Epoch {epoch+1} Train Completed - Avg Loss: {final_loss:.4f}\\n")
        
        if do_eval:
            val_epoch_losses = []
            val_probe_mses = []
            
            bridge_val_iter = bridge_val_loader.load(split="val")
            so100_val_iter = so100_val_loader.load(split="val")
            
            pbar_val = tqdm(zip(bridge_val_iter, so100_val_iter), desc=f"Epoch {epoch+1} Validation", unit="batch")
            for bridge_val_batch, so100_val_batch in pbar_val:
                metrics_b_val = eval_step_fn(state, bridge_val_batch)
                metrics_s_val = eval_step_fn(state, so100_val_batch)
                
                avg_val_loss = (metrics_b_val["loss"] + metrics_s_val["loss"]) / 2.0
                avg_val_probe_mse = (metrics_b_val["probe_loss"] + metrics_s_val["probe_loss"]) / 2.0
                
                val_epoch_losses.append(avg_val_loss)
                val_probe_mses.append(avg_val_probe_mse)
                
                pbar_val.set_postfix({
                    "Val Avg L": f"{avg_val_loss:.3f}",
                    "Val Probe MSE": f"{avg_val_probe_mse:.3f}"
                })
                
            final_val_loss = np.mean(val_epoch_losses)
            final_val_probe = np.mean(val_probe_mses)
            print(f"\\n🎯 Epoch {epoch+1} Validation Completed - Val Avg Loss: {final_val_loss:.4f} | Val Probe MSE: {final_val_probe:.4f}\\n")
            
            # If evaluating, return the validation loss to SMAC so we optimize for generalization!
            final_loss = final_val_loss
            
    # Return loss for SMAC3 Pareto evaluation
    return float(final_loss)
