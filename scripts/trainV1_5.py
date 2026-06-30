import argparse
import os
# Enable XLA Latency Hiding Scheduler for maximum overlapping of computation and memory
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_latency_hiding_scheduler=true"
import sys

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

def run_v1_5_burst(do_eval: bool = True, num_epochs: int = 20):
    """Executes the High-Acuity V1.5 Burst run at 512x512 resolution."""
    config = {
        "latent_dim": 256,         
        "vit_depth": 4,            
        "patch_size": 16,          
        "image_size": 512,
        "masking_ratio": 0.75,     
        "wm_depth": 4,             
        "num_heads": 16,           
        "batch_size": 16,          # Reduced dramatically to 16 to prevent OOM with 512x512 + Deep Supervision
        "seq_len": 6,              
        "activation_fn": "gelu",   
        "learning_rate": 0.0003,
        "probe_learning_rate": 0.004,
        "weight_decay": 0.005,
        "tau": 0.995,
        "loss_alpha": 1.0,         
        "sigreg_weight": 0.02,     
        "use_amp": True,           
        "disable_wandb": True,
        "sample_fraction": 1.0,
    }
    print(f"Running V1.5 High-Acuity Burst for {num_epochs} epochs. 512x512 Resolution mode.")
    final_loss = train_model(config, num_epochs=num_epochs, do_eval=do_eval, save_dir="checkpoints/v1_5_jepa_backbone")
    print(f"\\nV1.5 Burst run completed. Final Loss: {final_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JEPA Robotics V1.5 Training Orchestrator")
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
        help="Number of epochs to train (default: 20)."
    )
    
    args = parser.parse_args()
    run_v1_5_burst(do_eval=args.eval, num_epochs=args.epochs)
