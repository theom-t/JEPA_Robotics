import jax
import jax.numpy as jnp
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, EqualsCondition
from smac import HyperparameterOptimizationFacade, Scenario

def build_smac_scenario(run_name: str = "v1_jepa_world_model") -> Scenario:
    """
    Constructs the Hierarchical Configuration Space for SMAC3.
    This defines the bounds of our dynamic architecture optimization.
    """
    cs = ConfigurationSpace()
    
    # --- Perception Engine (V-JEPA) Hyperparameters ---
    latent_dim = Categorical("latent_dim", [128, 256, 512], default=256)
    vit_depth = Integer("vit_depth", (2, 8), default=4)
    patch_size = Categorical("patch_size", [16, 32, 64, 128], default=16)
    
    # V-JEPA Masking Hierarchy
    use_masking = Categorical("use_masking", [True, False], default=True)
    masking_ratio = Float("masking_ratio", (0.5, 0.9), default=0.7)
    
    # --- World Model Hyperparameters ---
    wm_depth = Integer("wm_depth", (2, 6), default=4)
    num_heads = Categorical("num_heads", [4, 8], default=8)
    
    # --- Temporal & Optimizer Dynamics ---
    tau = Float("tau", (0.99, 0.9999), default=0.996)
    learning_rate = Float("learning_rate", (1e-5, 1e-3), default=1e-4, log=True)
    weight_decay = Float("weight_decay", (1e-6, 1e-2), default=1e-4, log=True)
    batch_size = Categorical("batch_size", [8, 16, 32], default=32)
    seq_len = Integer("seq_len", (3, 10), default=5)
    activation_fn = Categorical("activation_fn", ["gelu", "silu", "relu"], default="gelu")
    loss_alpha = Float("loss_alpha", (0.1, 10.0), default=1.0)
    
    cs.add_hyperparameters([
        latent_dim, vit_depth, patch_size, 
        use_masking, masking_ratio,
        wm_depth, num_heads, 
        tau, learning_rate, weight_decay,
        batch_size, seq_len, activation_fn, loss_alpha
    ])
    
    # Inject SMAC Hierarchy: masking_ratio is ONLY active if use_masking is True
    cs.add_condition(EqualsCondition(masking_ratio, use_masking, True))
    
    # Define the Scenario
    # We want to minimize the combined loss (L2 Latent + Temporal Dynamics)
    scenario = Scenario(
        cs,
        deterministic=True, # JAX PRNG keys make evaluation deterministic
        n_trials=50,       # Number of architectures to evaluate
        name=run_name,
        output_directory="smac3_output", # Explicitly force the automatic JSON save behavior
        min_budget=4,      # Multi-fidelity min epochs
        max_budget=10      # Multi-fidelity max epochs
    )
    
    return scenario

from ConfigSpace import Configuration

def get_evaluation_function(do_eval: bool):
    # Notice we now accept `budget` from SMAC for multi-fidelity tuning
    def evaluation_function(config: Configuration, seed: int = 0, budget: float = 10.0) -> float:
        """
        The function SMAC calls to evaluate a given architecture.
        The 'budget' parameter tells us how many epochs to train for this specific trial.
        """
        from jepa_robotics.training.loop import train_model
        
        # Convert SMAC config to a standard dictionary
        config_dict = dict(config)
        config_dict["is_smac_run"] = True
        
        # Train for the number of epochs specified by the Hyperband budget
        epochs_to_run = int(budget)
        print(f"\n[SMAC] Evaluating config for {epochs_to_run} epochs (Budget: {budget})")
        
        loss = train_model(config_dict, num_epochs=epochs_to_run, do_eval=do_eval)
        return float(loss)
    return evaluation_function

def run_smac_optimization(do_eval: bool = True):
    """Executes the SMAC3 optimization loop with Hyperband."""
    from smac.intensifier.hyperband import Hyperband
    
    scenario = build_smac_scenario()
    
    # Enable Multi-Fidelity Hyperband Intensifier
    intensifier = Hyperband(
        scenario,
        incumbent_selection="highest_observed_budget",
        eta=2           # Changed to 2 so it halving perfectly (4 -> 8 -> 10)
    )
    
    # Use the Facade for standard Hyperparameter Optimization
    smac = HyperparameterOptimizationFacade(
        scenario, 
        get_evaluation_function(do_eval),
        intensifier=intensifier
    )
    
    incumbent = smac.optimize()
    print(f"Optimal Architecture Discovered: {incumbent}")
    return incumbent
