# Comprehensive Dataset Analysis Report
This report provides an in-depth breakdown of the robotic datasets currently available on disk for the Cross-Embodiment Latent World Model.

## 1. BridgeData V2 (Google / WidowX 250)
**Origin:** title={BridgeData V2: A Dataset for Robot Learning at Scale},
**Description:** WidowX interacting with toy kitchens
### 1.1 Data Volume & Sharding by Split
#### Split: `TRAIN`
- **Total Episodes Recorded:** 25,460
- **Total Disk Size (100%):** 340.85 GB
- **Current Shards Downloaded:** 1024 out of 1024 (100.0%)
- **Estimated Frames Available Locally:** ~891,100 frames
#### Split: `TEST`
- **Total Episodes Recorded:** 3,475
- **Total Disk Size (100%):** 46.65 GB
- **Current Shards Downloaded:** 512 out of 512 (100.0%)
- **Estimated Frames Available Locally:** ~121,625 frames
### 1.2 Schema & Feature Space
#### Observation Subspace:
- `image`: Camera feed. Shape: `(480, 640, 3)`. Dtype: `uint8`.
- `state`: Proprioceptive 7-DOF arm state. Shape: `(7,)`. Dtype: `float32`.
- `natural_language_instruction`: Human language task description. Dtype: `string`.
- `natural_language_embedding`: Pre-computed language embedding (USE/MUSE). Shape: `(512,)`. Dtype: `float32`.
#### Action Subspace:
- `world_vector`: End-effector XYZ translation. Shape: `(3,)`. Dtype: `float32`.
- `rotation_delta`: End-effector roll, pitch, yaw delta. Shape: `(3,)`. Dtype: `float32`.
- `open_gripper`: Gripper actuation state. Dtype: `boolean`.

---

## 2. LeRobot SO100 (Hugging Face / SO100 Stacking)
**Origin:** Hugging Face Hub (`lerobot/svla_so100_stacking`)
### 2.1 Data Volume & Format by Split
#### Split: `TRAIN`
- **Total Frames (Samples):** 22,956

### 2.2 Schema & Feature Space
Unlike TFDS which nests elements into 'episodes', the Hugging Face schema represents data as a continuous frame-by-frame flat table.
- `action`: List(Value('float32'))
- `observation.state`: List(Value('float32'))
- `timestamp`: Value('float32')
- `frame_index`: Value('int64')
- `episode_index`: Value('int64')
- `index`: Value('int64')
- `task_index`: Value('int64')
### 2.3 Empirical Statistics
Computed over the first 5,000 frames of `train`:
#### Proprioceptive State (`observation.state`)
- **Min Bound:** -19.2480
- **Max Bound:** 177.4512
- **Mean Magnitude:** 65.8887
- **Standard Deviation:** 52.2684
#### Action Space (`action`)
- **Min Bound:** -19.4238
- **Max Bound:** 177.2754
- **Mean Magnitude:** 65.3415
- **Standard Deviation:** 52.7755

---

## 3. Data Integration & Dataloader Alignment
To train the V-JEPA World Model across these fundamentally different robotic formats, the `dataset_loaders.py` pipeline forces them into a common vector space:
1. **Image Normalization:** BridgeData (480x640) and SO100 videos are both resized/cropped dynamically into `(256, 256, 3)`.
2. **Action Alignment:** BridgeData represents actions as separate `world_vector` and `rotation_delta` keys, whereas SO100 uses a unified array. The loaders mathematically pack/pad these into a universal `(7,)` DOF vector.
3. **Temporal Slicing:** Both streams are dynamically chunked into Sliding Windows of `seq_len=5`.