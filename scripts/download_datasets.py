import os
import argparse
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Download and cache robot datasets.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of episodes/frames downloaded for testing.")
    parser.add_argument("--data-dir", type=str, default="/home/tmainetucker/Repos/JEPA_Robotics/data", help="Local directory to store datasets.")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    
    print("=== JEPA Robotics: Decoupled Data Ingestion ===")
    print("Running in 'jepa_data' environment to avoid JAX/CUDA/Protobuf conflicts.\\n")
    
    # 1. Fetch BridgeData V2 via direct gsutil
    print("[1/2] Fetching BridgeData V2 (RLDS)...")
    bridge_dir = os.path.join(args.data_dir, "bridge_data_v2", "bridge", "0.1.0")
    os.makedirs(bridge_dir, exist_ok=True)
    
    # We use gsutil rsync with the -c (checksum) flag to directly sync the public bucket.
    # This prevents it from overwriting your 606 existing files just because their 'date modified' is newer!
    print(f"Synchronizing from gs://gresearch/robotics/bridge/0.1.0/ to {bridge_dir} ...")
    try:
        subprocess.run(["gsutil", "-m", "rsync", "-c", "-r", "gs://gresearch/robotics/bridge/0.1.0/", bridge_dir], check=True)
        print("BridgeData V2 cache complete.")
    except FileNotFoundError:
        print("ERROR: 'gsutil' is not installed. Please run: pip install gsutil")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: gsutil sync failed with code {e.returncode}")
        sys.exit(1)
        
    # 2. Fetch SO100 Data via Hugging Face
    print("\\n[2/2] Fetching SO100 Data (Hugging Face)...")
    try:
        from datasets import load_dataset
        so100_dir = os.path.join(args.data_dir, "lerobot_so100")
        os.makedirs(so100_dir, exist_ok=True)
        
        # Load the dataset to trigger the offline cache mechanism
        print("Downloading SO100 split='train'...")
        load_dataset("lerobot/svla_so100_stacking", split="train", cache_dir=so100_dir)
        print("SO100 cache complete.")
    except ImportError:
        print("ERROR: 'datasets' is not installed. Make sure you are in the 'jepa_data' conda environment!")
        sys.exit(1)

if __name__ == "__main__":
    main()
