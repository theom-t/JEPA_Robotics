#!/bin/bash
set -e

# Target SSD directory
DATA_DIR="/home/tmainetucker/Repos/JEPA_Robotics/data"

echo "=========================================================="
echo " JEPA Robotics - Massive Dataset Synchronization Engine"
echo " Target SSD: $DATA_DIR"
echo "=========================================================="

mkdir -p "$DATA_DIR/lerobot_so100"
mkdir -p "$DATA_DIR/bridge_data_v2"

echo ""
echo "[1/2] Syncing LeRobot SO100 (Hugging Face Hub)"
echo "----------------------------------------------------------"
# We use the new hf CLI and pull it directly into the local data folder
hf download lerobot/svla_so100_stacking --repo-type dataset --local-dir "$DATA_DIR/lerobot_so100"
echo "SO100 Download Complete."

echo ""
echo "[2/2] Syncing BridgeData V2 (Google Cloud Storage / RLDS)"
echo "----------------------------------------------------------"
# We use the new blazing fast Go-based gcloud storage CLI
# Note: The gresearch/robotics bucket is public.
/home/tmainetucker/google-cloud-sdk/bin/gcloud storage cp -R gs://gresearch/robotics/bridge "$DATA_DIR/bridge_data_v2"
echo "BridgeData V2 Download Complete."

echo ""
echo "=========================================================="
echo " All Datasets Successfully Synced to SSD."
echo "=========================================================="
