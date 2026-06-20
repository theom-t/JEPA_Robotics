import os
import json
import glob
from datasets import load_dataset
import numpy as np

def generate_report():
    docs_dir = "/home/tmainetucker/Repos/JEPA_Robotics/docs"
    os.makedirs(docs_dir, exist_ok=True)
    report_path = os.path.join(docs_dir, "dataset_analysis_report.md")
    
    report = []
    report.append("# Comprehensive Dataset Analysis Report")
    report.append("This report provides an in-depth breakdown of the robotic datasets currently available on disk for the Cross-Embodiment Latent World Model.\n")
    
    # ==========================================
    # BRIDGE DATA V2
    # ==========================================
    report.append("## 1. BridgeData V2 (Google / WidowX 250)")
    tfds_dir = "/home/tmainetucker/Repos/JEPA_Robotics/data/bridge_data_v2/bridge/0.1.0"
    if os.path.exists(tfds_dir):
        with open(os.path.join(tfds_dir, "dataset_info.json"), "r") as f:
            info = json.load(f)
            
        report.append(f"**Origin:** {info['citation'].splitlines()[1].strip()}")
        report.append(f"**Description:** {info['description']}")
        
        report.append("### 1.1 Data Volume & Sharding by Split")
        for split_info in info.get('splits', []):
            split_name = split_info['name']
            
            if 'statistics' in split_info and 'numExamples' in split_info['statistics']:
                total_episodes = int(split_info['statistics']['numExamples'])
            elif 'shardLengths' in split_info:
                total_episodes = sum(int(x) for x in split_info['shardLengths'])
            else:
                total_episodes = 0
                
            total_shards = len(split_info.get('shardLengths', []))
            total_bytes = int(split_info.get('numBytes', 0))
            
            # Count downloaded shards specifically for this split
            search_pattern = os.path.join(tfds_dir, f"bridge-{split_name}.tfrecord*")
            downloaded_shards = len(glob.glob(search_pattern))
            download_pct = (downloaded_shards / total_shards) * 100 if total_shards > 0 else 0
            
            report.append(f"#### Split: `{split_name.upper()}`")
            report.append(f"- **Total Episodes Recorded:** {total_episodes:,}")
            report.append(f"- **Total Disk Size (100%):** {total_bytes / (1024**3):.2f} GB")
            report.append(f"- **Current Shards Downloaded:** {downloaded_shards} out of {total_shards} ({download_pct:.1f}%)")
            if total_shards > 0:
                report.append(f"- **Estimated Frames Available Locally:** ~{int((total_episodes * 35) * (downloaded_shards / total_shards)):,} frames")
        
        report.append("### 1.2 Schema & Feature Space")
        with open(os.path.join(tfds_dir, "features.json"), "r") as f:
            feat = json.load(f)["featuresDict"]["features"]["steps"]["sequence"]["feature"]["featuresDict"]["features"]
            
        obs = feat["observation"]["featuresDict"]["features"]
        act = feat["action"]["featuresDict"]["features"]
        
        report.append("#### Observation Subspace:")
        img_shape = obs["image"]["image"]["shape"]["dimensions"]
        report.append(f"- `image`: Camera feed. Shape: `({img_shape[0]}, {img_shape[1]}, {img_shape[2]})`. Dtype: `{obs['image']['image']['dtype']}`.")
        report.append(f"- `state`: Proprioceptive 7-DOF arm state. Shape: `(7,)`. Dtype: `float32`.")
        report.append(f"- `natural_language_instruction`: Human language task description. Dtype: `string`.")
        report.append(f"- `natural_language_embedding`: Pre-computed language embedding (USE/MUSE). Shape: `(512,)`. Dtype: `float32`.")
        
        report.append("#### Action Subspace:")
        report.append(f"- `world_vector`: End-effector XYZ translation. Shape: `(3,)`. Dtype: `float32`.")
        report.append(f"- `rotation_delta`: End-effector roll, pitch, yaw delta. Shape: `(3,)`. Dtype: `float32`.")
        report.append(f"- `open_gripper`: Gripper actuation state. Dtype: `boolean`.")
    else:
        report.append("BridgeData V2 directory not found.")

    report.append("\n---\n")

    # ==========================================
    # LEROBOT SO100
    # ==========================================
    report.append("## 2. LeRobot SO100 (Hugging Face / SO100 Stacking)")
    hf_repo = "lerobot/svla_so100_stacking"
    offline_dir = "/home/tmainetucker/Repos/JEPA_Robotics/data/lerobot_so100"
    
    try:
        # Load dataset dictionary offline (contains all splits)
        ds_dict = load_dataset(hf_repo, cache_dir=offline_dir)
        report.append(f"**Origin:** Hugging Face Hub (`{hf_repo}`)")
        
        report.append("### 2.1 Data Volume & Format by Split")
        for split_name, split_ds in ds_dict.items():
            report.append(f"#### Split: `{split_name.upper()}`")
            report.append(f"- **Total Frames (Samples):** {len(split_ds):,}")
            
        report.append("\n### 2.2 Schema & Feature Space")
        # Grab first split to report features
        first_split = list(ds_dict.values())[0]
        report.append("Unlike TFDS which nests elements into 'episodes', the Hugging Face schema represents data as a continuous frame-by-frame flat table.")
        for feature_name, feature_def in first_split.features.items():
            report.append(f"- `{feature_name}`: {feature_def}")
            
        report.append("### 2.3 Empirical Statistics")
        sample_size = min(len(first_split), 5000)
        sample_states = np.array(first_split[:sample_size]["observation.state"])
        sample_actions = np.array(first_split[:sample_size]["action"])
        
        report.append(f"Computed over the first {sample_size:,} frames of `{list(ds_dict.keys())[0]}`:")
        report.append("#### Proprioceptive State (`observation.state`)")
        report.append(f"- **Min Bound:** {np.min(sample_states):.4f}")
        report.append(f"- **Max Bound:** {np.max(sample_states):.4f}")
        report.append(f"- **Mean Magnitude:** {np.mean(sample_states):.4f}")
        report.append(f"- **Standard Deviation:** {np.std(sample_states):.4f}")
        
        report.append("#### Action Space (`action`)")
        report.append(f"- **Min Bound:** {np.min(sample_actions):.4f}")
        report.append(f"- **Max Bound:** {np.max(sample_actions):.4f}")
        report.append(f"- **Mean Magnitude:** {np.mean(sample_actions):.4f}")
        report.append(f"- **Standard Deviation:** {np.std(sample_actions):.4f}")
        
    except Exception as e:
        report.append(f"Error reading LeRobot metadata: {e}")

    report.append("\n---\n")
    report.append("## 3. Data Integration & Dataloader Alignment")
    report.append("To train the V-JEPA World Model across these fundamentally different robotic formats, the `dataset_loaders.py` pipeline forces them into a common vector space:")
    report.append("1. **Image Normalization:** BridgeData (480x640) and SO100 videos are both resized/cropped dynamically into `(256, 256, 3)`.")
    report.append("2. **Action Alignment:** BridgeData represents actions as separate `world_vector` and `rotation_delta` keys, whereas SO100 uses a unified array. The loaders mathematically pack/pad these into a universal `(7,)` DOF vector.")
    report.append("3. **Temporal Slicing:** Both streams are dynamically chunked into Sliding Windows of `seq_len=5`.")

    with open(report_path, "w") as f:
        f.write("\n".join(report))
        
    print(f"✅ In-depth dataset report generated at: {report_path}")

if __name__ == "__main__":
    generate_report()
