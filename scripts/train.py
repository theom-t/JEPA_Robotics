import argparse
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from jepa_robotics.training.loop import train_model
from jepa_robotics.training.optimization import run_smac_optimization

def run_single_mode(do_eval: bool = True):
    """Executes a single debug/manual training run with predefined hyperparameters."""
    config = {
        "latent_dim": 256,
        "vit_depth": 4,
        "patch_size": 16,
        "use_masking": True,
        "masking_ratio": 0.5,
        "wm_depth": 4,
        "num_heads": 8,
        "batch_size": 16,
        "seq_len": 5,
        "activation_fn": "gelu",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "tau": 0.996,
        "loss_alpha": 1.0,
        "disable_wandb": True # Disable logging for simple tests, switch to False for real telemetry
    }
    print("Running in SINGLE mode. Using default hyperparameter config.")
    final_loss = train_model(config, num_epochs=10, do_eval=do_eval)
    print(f"\\nSingle run completed. Final Loss: {final_loss:.4f}")

def run_optimize_mode(do_eval: bool = True):
    """Executes the SMAC3 Optimization Loop to discover the Pareto Front."""
    print("Running in OPTIMIZE mode via SMAC3.")
    incumbent = run_smac_optimization(do_eval=do_eval)
    print("\\nOptimization Sweep Complete.")
    print(f"Discovered Optimal Configuration: {incumbent}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JEPA Robotics Dual-Mode Training Orchestrator")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["single", "optimize"], 
        required=True,
        help="Choose 'single' for a fast manual run or 'optimize' to trigger SMAC3."
    )
    parser.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the validation loop on a held-out test split after each epoch."
    )
    
    args = parser.parse_args()
    
    if args.mode == "single":
        run_single_mode(do_eval=args.eval)
    elif args.mode == "optimize":
        run_optimize_mode(do_eval=args.eval)
