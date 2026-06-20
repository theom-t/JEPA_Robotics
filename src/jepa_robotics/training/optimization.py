import jax
import jax.numpy as jnp
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical
from smac import HyperparameterOptimizationFacade, Scenario

def build_smac_scenario(run_name: str = "v1_jepa_world_model") -> Scenario:
    """
    Constructs the Hierarchical Configuration Space for SMAC3.
    This defines the bounds of our dynamic architecture optimization.
    """
    cs = ConfigurationSpace()
    
    # --- Perception Engine (V-JEPA) Hyperparameters ---
    # The size of the dense latent vector we compress images into.
    # 256 is fast for Jetson Orin Nano, 512 is better for complex physics.
    latent_dim = Categorical("latent_dim", [128, 256, 512], default=256)
    
    # Depth of the Vision Transformer
    vit_depth = Integer("vit_depth", (2, 8), default=4)
    
    # --- World Model Hyperparameters ---
    wm_depth = Integer("wm_depth", (2, 6), default=4)
    
    # We must ensure num_heads cleanly divides latent_dim.
    # For a categorical latent_dim, we can just use 4 or 8 heads since they 
    # divide 128, 256, and 512 cleanly.
    num_heads = Categorical("num_heads", [4, 8], default=8)
    
    # --- JEPA Training Dynamics ---
    # EMA update rate (tau) for the Target Encoder.
    # Closer to 1.0 means slower, more stable target updates.
    tau = Float("tau", (0.99, 0.9999), default=0.996)
    
    # Learning rate
    learning_rate = Float("learning_rate", (1e-5, 1e-3), default=1e-4, log=True)
    
    cs.add_hyperparameters([latent_dim, vit_depth, wm_depth, num_heads, tau, learning_rate])
    
    # Define the Scenario
    # We want to minimize the combined loss (L2 Latent + Temporal Dynamics)
    scenario = Scenario(
        cs,
        deterministic=True, # JAX PRNG keys make evaluation deterministic
        n_trials=50,        # Number of architectures to evaluate
        name=run_name,
        output_directory="smac3_output" # Explicitly force the automatic JSON save behavior
    )
    
    return scenario

def evaluation_function(config, seed: int = 0) -> float:
    """
    Executes the real JAX training loop via our dual-mode orchestrator.
    Returns the final validation loss.
    """
    # Convert ConfigSpace object to a standard dict for our orchestrator
    config_dict = dict(config)
    config_dict["disable_wandb"] = False # Ensure telemetry runs during SMAC3
    
    from jepa_robotics.training.loop import train_model
    
    # We use a short number of epochs (e.g., 2) for the SMAC3 sweep evaluation 
    # to find the pareto front quickly.
    loss = train_model(config_dict, num_epochs=2)
    return float(loss)

def run_smac_optimization():
    """Executes the SMAC3 optimization loop."""
    scenario = build_smac_scenario()
    
    # Use the Facade for standard Hyperparameter Optimization
    smac = HyperparameterOptimizationFacade(
        scenario, 
        evaluation_function
    )
    
    incumbent = smac.optimize()
    print(f"Optimal Architecture Discovered: {incumbent}")
    return incumbent
