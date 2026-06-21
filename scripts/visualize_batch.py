import matplotlib.pyplot as plt
import numpy as np
from jepa_robotics.data.dataset_loaders import SO100DataLoader, BridgeDataLoader

def visualize_batch(loader, dataset_name: str):
    print(f"\nVisualizing batch for {dataset_name}...")
    data_iterator = loader.load()
    
    limit = loader.limit if loader.limit is not None else 5
    
    fig, axes = plt.subplots(1, limit, figsize=(5 * limit, 5))
    if limit == 1:
        axes = [axes]
        
    for i in range(limit):
        try:
            batch = next(data_iterator)
        except StopIteration:
            break
            
        # The image is expected to be [B, S, H, W, C] due to the temporal sliding window
        # We grab the first batch item (0), and the first frame in the sequence (0)
        img = np.array(batch["image"][0, 0])
        axes[i].imshow(img)
        axes[i].set_title(f"Frame {i}")
        
        cartesian = batch.get("state_7d", None)
        if cartesian is not None:
            xyz = cartesian[0, 0][:3]
            axes[i].text(10, 20, f"X:{xyz[0]:.2f} Y:{xyz[1]:.2f} Z:{xyz[2]:.2f}", 
                        color='white', backgroundcolor='black')
        
        axes[i].axis('off')
                
    plt.suptitle(f"{dataset_name} Data Pipeline Check")
    plt.tight_layout()
    
    save_path = f"visualize_{dataset_name.lower()}.png"
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    print("Testing data pipelines...\n")
    
    # Initialize loaders
    bridge_loader = BridgeDataLoader(
        limit=15
    )
    
    so100_loader = SO100DataLoader(
        hf_repo="lerobot/svla_so100_stacking",
        limit=15
    )
    
    visualize_batch(bridge_loader, "BridgeData_WidowX")
    visualize_batch(so100_loader, "SO100")
