#!/usr/bin/env bash
# Step 0: Process raw NAVSIM data into the DriveDreamer-Policy format.
# Run: source env.sh && bash 0-process_data.sh

set -euo pipefail

SPLIT=mini
DATA_ROOT=navsim_dataset
# Add --make_video to generate video clips; omit it to skip and speed up processing

CUDA_VISIBLE_DEVICES=0 python navsim_data_process/make_data.py \
    --split "$SPLIT" \
    --data_root "$DATA_ROOT"
