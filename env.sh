#!/usr/bin/env bash
# =============================================================================
# DriveDreamer-Policy — User Environment Configuration
# Copy this file and fill in the paths for your own setup.
# Then source it before running any of the numbered pipeline scripts:
#   source env.sh
# =============================================================================

# ── CUDA ──────────────────────────────────────────────────────────────────────
export CUDA_HOME=/usr/local/cuda-12.4          # adjust to your CUDA path
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ── HuggingFace / Torch cache ─────────────────────────────────────────────────
export HF_HOME=$HOME/.cache/huggingface
export TORCH_HOME=$HF_HOME
# Uncomment if you use a mirror (e.g. in mainland China):
# export HF_ENDPOINT=https://hf-mirror.com

# ── NAVSIM / nuPlan paths ─────────────────────────────────────────────────────
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/navsim_dataset/maps"   # maps downloaded from NAVSIM
export NAVSIM_EXP_ROOT="navsim_exp"                      # output directory for experiments
export NAVSIM_DEVKIT_ROOT="$(pwd)/navsim"              # cloned navsim devkit (v2)
export OPENSCENE_DATA_ROOT="/path/to/navsim_dataset"     # root of the NAVSIM dataset

# ── Base VLM checkpoint ───────────────────────────────────────────────────────
# Output of step 7 (7-add_token.sh): Qwen3-VL-2B-Instruct with extended world & action tokens
export BASE_VLM=/path/to/Qwen3-VL-2B-WorldAction

# ── Weights & Biases (optional) ───────────────────────────────────────────────
export WANDB_API_KEY=your_wandb_api_key_here
export WANDB_ENTITY=your_wandb_entity
export WANDB_PROJECT=drivedreamer-policy

# export WANDB_MODE=offline