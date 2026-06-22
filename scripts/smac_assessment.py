import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    runhistory_path = "/home/tmainetucker/Repos/JEPA_Robotics/smac3_output/v1_jepa_world_model/0/runhistory.json"
    output_dir = "/home/tmainetucker/Repos/JEPA_Robotics/smac3_output/graphs"
    
    if not os.path.exists(runhistory_path):
        print(f"Runhistory not found at {runhistory_path}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading runhistory from {runhistory_path}...")
    
    with open(runhistory_path, 'r') as f:
        history = json.load(f)
        
    data = history.get("data", [])
    configs = history.get("configs", {})
    
    # Track the best cost over time
    trials = []
    costs = []
    best_costs = []
    current_best = float('inf')
    
    # Store hyperparams for scatter plots
    param_data = {
        'learning_rate': [],
        'probe_learning_rate': [],
        'masking_ratio': [],
        'vit_depth': [],
        'cost': []
    }
    
    for i, entry in enumerate(data):
        config_id = str(entry["config_id"])
        cost = entry["cost"]
        budget = entry["budget"]
        
        if cost > 1.0:
            print(f"Skipping outlier config {config_id} (Cost: {cost:.2f})")
            continue
        
        # Only consider fully evaluated configs or just plot everything
        if cost < current_best:
            current_best = cost
            
        trials.append(i + 1)
        costs.append(cost)
        best_costs.append(current_best)
        
        # Extract hyperparams
        cfg = configs.get(config_id, {})
        param_data['learning_rate'].append(cfg.get('learning_rate', np.nan))
        param_data['probe_learning_rate'].append(cfg.get('probe_learning_rate', np.nan))
        param_data['masking_ratio'].append(cfg.get('masking_ratio', 0.0))
        param_data['vit_depth'].append(cfg.get('vit_depth', np.nan))
        param_data['cost'].append(cost)
        
    print(f"Parsed {len(trials)} trials. Current best cost: {current_best:.4f}")
    
    # Plot 1: Optimization Trace (Convergence)
    plt.figure(figsize=(10, 6))
    plt.scatter(trials, costs, alpha=0.5, label='Individual Trial Cost')
    plt.plot(trials, best_costs, color='red', linewidth=2, label='Best Cost Found')
    plt.xlabel("Trial Number")
    plt.ylabel("SMAC Score (Weighted Probe MSE)")
    plt.title("SMAC Optimization Trajectory")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimization_trace.png"))
    plt.close()
    
    # Helper to plot parameter scatters
    def plot_scatter(param_name, log_x=False):
        plt.figure(figsize=(8, 6))
        x = param_data[param_name]
        y = param_data['cost']
        
        # Filter NaNs
        valid = [i for i in range(len(x)) if not np.isnan(x[i])]
        x_clean = [x[i] for i in valid]
        y_clean = [y[i] for i in valid]
        
        if not x_clean:
            return
            
        plt.scatter(x_clean, y_clean, alpha=0.7)
        if log_x:
            plt.xscale('log')
            
        # Add best point star
        best_idx = np.argmin(y_clean)
        plt.scatter([x_clean[best_idx]], [y_clean[best_idx]], color='red', marker='*', s=200, label='Best Config')
        
        plt.xlabel(param_name)
        plt.ylabel("SMAC Score")
        plt.title(f"Effect of {param_name} on Performance")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scatter_{param_name}.png"))
        plt.close()
        
    # Plot scatters
    plot_scatter('learning_rate', log_x=True)
    plot_scatter('probe_learning_rate', log_x=True)
    plot_scatter('masking_ratio', log_x=False)
    plot_scatter('vit_depth', log_x=False)
    
    print(f"Successfully generated 5 assessment graphs in {output_dir}")

if __name__ == "__main__":
    main()
