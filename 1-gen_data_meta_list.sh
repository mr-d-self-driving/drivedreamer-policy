#!/usr/bin/env bash
# Step 1: Generate the data meta-list JSON used by the dataloader.
# Run: source env.sh && bash 1-gen_data_meta_list.sh

set -euo pipefail

SPLIT=mini
DATA_ROOT=navsim_dataset

CUDA_VISIBLE_DEVICES=0 python navsim_data_process/data_list.py \
    --split "$SPLIT" \
    --data_root "$DATA_ROOT"
