import jax
import jax.numpy as jnp
import optax
import wandb
import numpy as np
from tqdm import tqdm
from typing import Dict, Any

from jepa_robotics.models.v_jepa import ViTEncoder, JEPAPredictor
from jepa_robotics.models.world_model import ActionConditionedTransformer
from jepa_robotics.training.step import create_train_step
from jepa_robotics.data.dataset_loaders import BridgeDataLoader, SO100DataLoader

def train_model(config: Dict[str, Any], num_epochs: int = 1) -> float:
    """
    Core orchestrator that binds dataloaders, compiles the network, 
    and executes the training loop.
    Returns the final validation loss.
    """
    # 1. Initialize Telemetry
    # We use reinit=True so SMAC3 can run multiple trials sequentially in the same process
    run = wandb.init(
        project="JEPA_Robotics",
        config=config,
        reinit=True,
        mode="disabled" if config.get("disable_wandb", False) else "online"
    )
    
    # 2. Instantiate Dynamic Architectures
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    latent_dim = config["latent_dim"]
    vit_depth = config["vit_depth"]
    wm_depth = config["wm_depth"]
    num_heads = config["num_heads"]
    lr = config["learning_rate"]
    tau = config["tau"]
    
    encoder_def = ViTEncoder(latent_dim=latent_dim, depth=vit_depth, num_heads=num_heads)
    predictor_def = JEPAPredictor(latent_dim=latent_dim)
    wm_def = ActionConditionedTransformer(latent_dim=latent_dim, depth=wm_depth, num_heads=num_heads)
    
    # Mock inputs for initialization
    mock_img = jnp.ones((1, 256, 256, 3))
    mock_seq_latents = jnp.ones((1, 1, latent_dim))
    mock_seq_actions = jnp.ones((1, 1, 7))
    
    # Initialize parameters
    encoder_params = encoder_def.init(init_rng, mock_img)
    target_params = encoder_def.init(init_rng, mock_img) # Clone structure
    predictor_params = predictor_def.init(init_rng, jnp.ones((1, latent_dim)))
    wm_params = wm_def.init(init_rng, mock_seq_latents, mock_seq_actions)
    
    # 3. Setup Optax Optimizer
    optimizer = optax.adamw(learning_rate=lr)
    
    params_tuple = (encoder_params, predictor_params, wm_params)
    opt_state = optimizer.init(params_tuple)
    
    # Pack into state dictionary
    state = {
        "encoder_params": encoder_params,
        "predictor_params": predictor_params,
        "wm_params": wm_params,
        "target_params": target_params,
        "opt_state": opt_state,
        "optimizer": optimizer
    }
    
    # 4. Compile JAX Train Step
    train_step_fn = create_train_step(encoder_def, predictor_def, wm_def)
    
    # 5. Initialize Dataloaders
    # We use limits to keep the V1 mock training fast
    bridge_loader = BridgeDataLoader(limit=10)
    so100_loader = SO100DataLoader(hf_repo="lerobot/svla_so100_stacking", limit=10)
    
    final_loss = 0.0
    
    # 6. Execute Alternating Training Loop
    print(f"\\nStarting Training Run (Latent: {latent_dim}, Epochs: {num_epochs})...")
    for epoch in range(num_epochs):
        bridge_iter = bridge_loader.load()
        so100_iter = so100_loader.load()
        
        # We zip them to alternate batches cleanly
        epoch_losses = []
        for bridge_batch, so100_batch in zip(bridge_iter, so100_iter):
            # Process BridgeData
            state, metrics_b = train_step_fn(state, bridge_batch, tau)
            
            # Process SO100 (Cross-Embodiment sharing the SAME state/weights!)
            state, metrics_s = train_step_fn(state, so100_batch, tau)
            
            # Log telemetry
            avg_loss = (metrics_b["loss"] + metrics_s["loss"]) / 2.0
            epoch_losses.append(avg_loss)
            
            wandb.log({
                "Bridge_Latent_Loss": metrics_b["latent_l2_loss"],
                "SO100_Latent_Loss": metrics_s["latent_l2_loss"],
                "Bridge_Temporal_Loss": metrics_b["temporal_dynamics_loss"],
                "SO100_Temporal_Loss": metrics_s["temporal_dynamics_loss"],
                "Combined_Loss": avg_loss
            })
            
        final_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {final_loss:.4f}")
        
    wandb.finish()
    
    # Return loss for SMAC3 Pareto evaluation
    return float(final_loss)
