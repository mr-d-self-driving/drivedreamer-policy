#!/usr/bin/env bash
# Step 5: Evaluate predictions on NAVSIM v1.1 (PDM-Score).
# Requires a separate conda environment: conda activate navsim_v1.1
# Run: conda activate navsim_v1.1 && source env.sh && bash 5-eval_v1.sh

set -euo pipefail

: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT in env.sh}"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT in env.sh}"

SPLIT=test
PRED_DIR=$(pwd)/navsim_planning_results/your-run-id  # /path/to/model results
METRIC_CACHE_PATH=/path/to/metric_cache_${SPLIT}    # pre-computed via run_metric_caching.sh

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="$(pwd)/navsim_exp/eval_v1.1"
export NAVSIM_DEVKIT_ROOT="$(pwd)/navsim_v1.1/navsim"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"
export SPLIT PRED_DIR METRIC_CACHE_PATH

# if not cached navsim data, run the following first
# cd navsim_v1.1/navsim/scripts/evaluation/
# CUDA_VISIBLE_DEVICES=0  ./run_metric_caching.sh

cd navsim_v1.1/navsim/scripts/evaluation/
CUDA_VISIBLE_DEVICES=0 ./run_human_agent_pdm_score_evaluation.sh
