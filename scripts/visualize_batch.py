import matplotlib.pyplot as plt
import numpy as np
from jepa_robotics.data.dataset_loaders import SO100DataLoader, BridgeDataLoader

def visualize_batch(loader, robot_name: str, num_samples: int = 3):
    """Pulls a batch from the loader and visualizes it."""
    print(f"\nVisualizing batch for {robot_name}...")
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    if num_samples == 1:
        axes = [axes]
        
    data_iterator = loader.load()
    
    for i in range(num_samples):
        try:
            batch = next(data_iterator)
        except StopIteration:
            break
            
        # Extract the mock image and state (convert from JAX to NumPy for plotting)
        img = np.array(batch["image"][0])
        state_7d = np.array(batch["state_7d"][0])
        
        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')
        
        # Display the 7D Cartesian vector on the image
        text_str = f"X,Y,Z: {state_7d[0]:.2f}, {state_7d[1]:.2f}, {state_7d[2]:.2f}\n" \
                   f"R,P,Y: {state_7d[3]:.2f}, {state_7d[4]:.2f}, {state_7d[5]:.2f}\n" \
                   f"Gripper: {state_7d[6]:.2f}"
                   
        ax.set_title(f"Sample {i+1}", fontsize=10)
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
    plt.suptitle(f"{robot_name} Data Pipeline Check")
    plt.tight_layout()
    
    save_path = f"visualize_{robot_name.lower()}.png"
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    # Test with a very small limit
    print("Testing data pipelines...")
    
    bridge_loader = BridgeDataLoader(limit=3)
    visualize_batch(bridge_loader, "BridgeData_WidowX")
    
    so100_loader = SO100DataLoader(limit=3)
    visualize_batch(so100_loader, "SO100")
