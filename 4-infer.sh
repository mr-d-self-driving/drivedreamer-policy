#!/usr/bin/env bash
# Step 4: Run inference with a trained checkpoint and save per-token .npy predictions.
# Run: source env.sh && bash 4-infer.sh

set -euo pipefail

# ── Required env vars (set in env.sh) ─────────────────────────────────────────
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT in env.sh}"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT in env.sh}"

# ── Configuration ─────────────────────────────────────────────────────────────
SPLIT=mini        # test | navhard_two_stage
DATA_ROOT=navsim_dataset
MODEL_DIR=/path/to/your/model_checkpoint  # e.g. ${NAVSIM_EXP_ROOT}/your-run-id

set -x
pwd

CUDA_VISIBLE_DEVICES=0 python infer.py \
  --ckpt_dir "${MODEL_DIR}" \
  --datalist_path "${SPLIT}_meta.json" \
  --data_root "${DATA_ROOT}" \
  --out_dir navsim_planning_results/ \
  --split "${SPLIT}" \
  --batch_size 8 \
  --num_workers 7 \
  --overwrite \
  --smooth 0
