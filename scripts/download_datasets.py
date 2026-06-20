import os
import argparse
from jepa_robotics.data.dataset_loaders import BridgeDataLoader, SO100DataLoader

def main():
    parser = argparse.ArgumentParser(description="Download and cache robot datasets.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of episodes/frames downloaded for testing.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Local directory to store datasets.")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    
    print("=== JEPA Robotics: Data Ingestion ===")
    
    # Initialize Loaders
    bridge_loader = BridgeDataLoader(data_dir=args.data_dir, limit=args.limit)
    so100_loader = SO100DataLoader(data_dir=args.data_dir, limit=args.limit)
    
    # Trigger load to cache datasets locally
    print("\n[1/2] Fetching BridgeData V2 (RLDS)...")
    for batch in bridge_loader.load():
        pass # In a real implementation, the HF/TFDS backend handles caching on the first call.
    print("BridgeData V2 cache complete.")
        
    print("\n[2/2] Fetching SO100 Data (Hugging Face)...")
    for batch in so100_loader.load():
        pass
    print("SO100 cache complete.")

if __name__ == "__main__":
    main()
