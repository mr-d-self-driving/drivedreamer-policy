#!/usr/bin/env bash
# Step 8 (train): Multi-GPU training with DeepSpeed ZeRO-2.
# Prerequisites:
#   1. source env.sh           (sets NAVSIM_EXP_ROOT, BASE_VLM, WANDB_*, …)
#   2. Prepare data (steps 0-3) and generate the meta-list JSON.
# Run: source env.sh && bash 6-train.sh

set -euo pipefail

# ── Required env vars (set in env.sh) ─────────────────────────────────────────
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT in env.sh}"
: "${BASE_VLM:?Set BASE_VLM in env.sh}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY in env.sh}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT in env.sh}"

# ── Experiment ID ─────────────────────────────────────────────────────────────
debug=false
if [ "${debug,,}" = "true" ]; then
  timestamp="debug"
else
  timestamp="$(date +"%m%d_%H")"
fi
echo "timestamp: $timestamp"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
num_processes=8
bz=4                  # per-device batch size
act_fm_size=1536      # action DiT hidden size
act_fm_layer=24       # action DiT number of layers
fm_repeat=8           # repeated diffusion steps

VIDEO_MODEL=/path/to/Wan2.1-Fun-V1.1-1.3B-InP/snapshots/xxx  # root directory of the downloaded model
VIDEO_CONFIG=starVLA/model/modules/video_model/config/wan2.1/wan_civitai.yaml
VIDEO_DATA_DIR=navsim_dataset/navsim_video  # /path/to/navsim_video

split=navtrain
datalist=${split}_meta.json

run_id=${timestamp}-3d-2d-1d-lr1e5-3d_loss_1e1-decay1e3-${split}_data-bz_${bz}_${num_processes}

Framework_name=QwenOFT
vl_hidden_dim=2048

export WANDB_MODE=offline

set -x
pwd

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --main_process_port 29687 \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/cfg_yaw_1225.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${BASE_VLM} \
  --framework.qwenvl.vl_hidden_dim ${vl_hidden_dim} \
  --run_root_dir ${NAVSIM_EXP_ROOT} \
  --run_id ${run_id} \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_entity ${WANDB_ENTITY} \
  --datasets.vla_data.datalist_path ${datalist} \
  --datasets.vla_data.split ${split} \
  --datasets.vla_data.per_device_batch_size ${bz} \
  --framework.action_model.repeated_diffusion_steps ${fm_repeat} \
  --datasets.video_data.load_2d_data 1 \
  --w_depth 1 \
  --gs_query_loss 1 \
  --rgb_query_loss 1 \
  --trainer.freeze_modules "rgb_model.vae,rgb_model.clip_image_encoder,rgb_model.text_encoder,qwen_vl_interface.model.visual" \
  --framework.action_model.hidden_size ${act_fm_size} \
  --framework.action_model.diffusion_model_cfg.cross_attention_dim ${act_fm_size} \
  --framework.action_model.diffusion_model_cfg.output_dim ${act_fm_size} \
  --framework.action_model.diffusion_model_cfg.num_layers ${act_fm_layer} \
  --trainer.optimizer.weight_decay 1e-3 \
  --trainer.learning_rate.base 1e-5 \
  --trainer.learning_rate.rgb_model 1e-5 \
  --framework.video_model.model_name ${VIDEO_MODEL} \
  --framework.video_model.config_path ${VIDEO_CONFIG} \
  --datasets.video_data.rgb_meta_dir ${VIDEO_DATA_DIR} \
  --trainer.max_train_steps 100000
