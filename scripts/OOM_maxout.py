import os
import sys

# Add src to path so we can import jepa_robotics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from jepa_robotics.training.loop import train_model

def run_oom_test():
    print("=" * 80)
    print("🚀 INITIATING OOM MAX-OUT STRESS TEST 🚀")
    print("=" * 80)
    print("Pushing all architecture and batch hyperparameters to their absolute limits to test RTX 5090 VRAM...")
    
    # Define the absolute maximum memory-hungry configuration based on our SMAC search space bounds
    max_config = {
        "latent_dim": 512,         # Max feature dimension
        "vit_depth": 8,            # Max encoder layers
        "patch_size": 16,           # MIN patch size = MAX sequence length for attention (1024 patches per image)
        "use_masking": True,       
        "masking_ratio": 0.5,      # Keeping 50% patches (lowest masking ratio = most patches kept in encoder)
        "wm_depth": 6,             # Max world model layers
        "num_heads": 8,            # Max attention heads
        "batch_size": 32,          # Max batch size
        "seq_len": 10,             # Max sequence length (10 frames per forward pass)
        "activation_fn": "gelu",
        "tau": 0.996,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "loss_alpha": 1.0,
        "disable_wandb": True,     # Disable telemetry for the stress test
        "is_smac_run": True        # Just use a small slice of data to test OOM, not the whole epoch
    }
    
    for key, value in max_config.items():
        print(f"  - {key}: {value}")
    print("=" * 80)
    
    try:
        # Run for just 1 epoch to trigger the heavy JIT compilation and memory allocation
        train_model(max_config, num_epochs=1)
        print("\n✅ SUCCESS! The RTX 5090 swallowed the maximum configuration without an OOM error!")
    except Exception as e:
        print(f"\n❌ OOM FATAL ERROR: The system crashed at the limits. You may need to constrain the SMAC space.\n")
        raise e

if __name__ == "__main__":
    # Force XLA to pre-allocate so we get a true out-of-memory crash if it exceeds physical bounds
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    run_oom_test()
