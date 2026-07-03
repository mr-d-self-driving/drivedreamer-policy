#!/usr/bin/env bash
# Step 6 (eval): Evaluate predictions on NAVSIM v2 (PDM-Score).
# Requires conda activate vla
# Run: conda activate vla && source env.sh && bash 6-eval_v2.sh

set -euo pipefail

: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT in env.sh}"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT in env.sh}"

SPLIT=test
PRED_DIR=$(pwd)/navsim_planning_results/your-run-id  # /path/to/model results
METRIC_CACHE_PATH=/path/to/metric_cache_${SPLIT}    # pre-computed via run_metric_caching.sh

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="$(pwd)/navsim_exp/eval_v2"
export NAVSIM_DEVKIT_ROOT="$(pwd)/navsim"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export SPLIT PRED_DIR METRIC_CACHE_PATH

# if not cached navsimv2 data, run the following first
# cd navsim/scripts/evaluation/
# CUDA_VISIBLE_DEVICES=0  ./run_metric_caching.sh

cd navsim/scripts/evaluation/
CUDA_VISIBLE_DEVICES=0 ./run_human_agent_pdm_score_evaluation.sh
