import argparse
import sys
import os

# Suppress XLA C++ warnings and prevent JAX from searching for TPUs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_PLATFORMS"] = "cuda,cpu"

# Force Hugging Face completely offline to prevent HTTP pings during cyclic dataloading
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from jepa_robotics.training.loop import train_model
from jepa_robotics.training.optimization import run_smac_optimization

def run_single_mode(do_eval: bool = True, num_epochs: int = 20, fast_test: bool = False):
    """Executes a single debug/manual training run with predefined hyperparameters."""
    config = {
        "latent_dim": 128,
        "vit_depth": 3,
        "patch_size": 32,
        "use_masking": True,
        "masking_ratio": 0.738,
        "wm_depth": 6,
        "num_heads": 4,
        "batch_size": 32,
        "seq_len": 5,
        "activation_fn": "relu",
        "learning_rate": 0.0005976,
        "probe_learning_rate": 0.0001663,
        "weight_decay": 0.002092,
        "tau": 0.9995,
        "loss_alpha": 6.457,
        "disable_wandb": True, # Disable logging for simple tests, switch to False for real telemetry
    }
    if fast_test:
        config["sample_fraction"] = 0.1 # 10% data for fast testing without batch caps
    print(f"Running in SINGLE mode for {num_epochs} epochs. Using default hyperparameter config.")
    final_loss = train_model(config, num_epochs=num_epochs, do_eval=do_eval, save_dir="checkpoints/v1_jepa")
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs to run when in 'single' mode (default: 100)."
    )
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run on a tiny 4% fraction of the data to quickly test the pipeline."
    )
    
    args = parser.parse_args()
    
    if args.mode == "single":
        run_single_mode(do_eval=args.eval, num_epochs=args.epochs, fast_test=args.fast)
    elif args.mode == "optimize":
        run_optimize_mode(do_eval=args.eval)
