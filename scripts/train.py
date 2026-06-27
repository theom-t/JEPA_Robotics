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

def run_single_mode(do_eval: bool = True, num_epochs: int = 20, fraction: float = 1.0):
    """Executes a single debug/manual training run with predefined hyperparameters."""
    config = {
        "latent_dim": 256,         # Jetson Orin Nano friendly, highest SMAC peak
        "vit_depth": 4,            # SMAC proved depth 4 is highly stable. 7 was causing gradient explosions.
        "patch_size": 16,          # High spatial acuity for robot manipulation
        "masking_ratio": 0.75,     # Strong semantic reasoning pressure
        "wm_depth": 4,             # Sufficient depth for temporal prediction
        "num_heads": 16,           # Best SMAC performer
        "batch_size": 128,         # Halved VRAM from bfloat16 AMP allows doubling batch size
        "seq_len": 6,              # Forced long-horizon temporal reasoning (No SMAC cheating)
        "activation_fn": "gelu",   # Best peak
        "learning_rate": 0.0003,
        "probe_learning_rate": 0.004,
        "weight_decay": 0.005,
        "tau": 0.995,
        "loss_alpha": 1.0,         # Stable L1/L2 weighting
        "sigreg_weight": 10.0,     # Increased to 10.0 (VICReg standard) to fiercely prevent Positional Collapse
        "use_amp": True,           # bfloat16 AMP re-enabled
        "disable_wandb": True,
        "sample_fraction": fraction,
    }
    print(f"Running in SINGLE mode for {num_epochs} epochs with {fraction*100:.0f}% data. Using final V-JEPA backbone config.")
    final_loss = train_model(config, num_epochs=num_epochs, do_eval=do_eval, save_dir="checkpoints/v1_jepa_backbone")
    print(f"\\nSingle run completed. Final Loss: {final_loss:.4f}")

def run_optimize_mode(do_eval: bool = True):
    """Executes the SMAC3 Optimization Loop to discover the Pareto Front."""
    print("Running in OPTIMIZE mode via SMAC3.")
    incumbent = run_smac_optimization(do_eval=do_eval)
    print("\nOptimization Sweep Complete.")
    print(f"Discovered Optimal Configuration: {incumbent}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JEPA Robotics Dual-Mode Training Orchestrator")
    parser.add_argument("--sigreg", type=float, default=10.0, help="SIGReg info-max weight")
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
        default=2, 
        help="Number of epochs to train (default: 2)."
    )
    parser.add_argument(
        "--fraction", 
        type=float, 
        default=1.0, 
        help="Fraction of data to use (e.g. 0.3 for 30%). Default is 1.0 (100%)."
    )
    
    args = parser.parse_args()
    
    if args.mode == "single":
        run_single_mode(do_eval=args.eval, num_epochs=args.epochs, fraction=args.fraction)
    elif args.mode == "optimize":
        run_optimize_mode(do_eval=args.eval)
