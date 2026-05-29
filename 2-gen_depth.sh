#!/usr/bin/env bash
# Step 2: Generate monocular depth maps for all scenes using Depth-Anything-3.
# Requires conda activate img_process before running.
# Run: source env.sh && bash 2-gen_depth.sh

set -euo pipefail

SPLIT=mini
DATA_ROOT=navsim_dataset
PROJECT_DIR=$(pwd)

export PYTHONPATH="$PROJECT_DIR/depth_process/Depth-Anything-3/src:$PYTHONPATH"
export HF_HUB_OFFLINE=1

cd depth_process
CUDA_VISIBLE_DEVICES=0 python depth.py \
    --split "$SPLIT" \
    --data_root "$PROJECT_DIR/$DATA_ROOT"
