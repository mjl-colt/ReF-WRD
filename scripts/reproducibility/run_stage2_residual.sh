#!/usr/bin/env bash
set -euo pipefail

: "${SD21BASE_PATH:?Set SD21BASE_PATH to the Stable Diffusion 2.1 base directory}"
: "${RAM_PATH:?Set RAM_PATH to the RAM checkpoint}"
: "${REFWRD_BASE_CKPT:?Set REFWRD_BASE_CKPT to the Stage-I checkpoint}"
: "${TRAIN_TXT:?Set TRAIN_TXT to the LSDIR training list}"
: "${FFHQ_TXT:?Set FFHQ_TXT to the FFHQ list}"

SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-experience/refwrd_stage2_seed${SEED}}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" accelerate launch train_fde_sr.py \
  --pretrained_model_name_or_path="${SD21BASE_PATH}" \
  --ram_path="${RAM_PATH}" \
  --resume_fde_sr_path="${REFWRD_BASE_CKPT}" \
  --train_stage2_only \
  --learning_rate=1e-4 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps 500 \
  --mixed_precision='fp16' \
  --report_to "tensorboard" \
  --seed "${SEED}" \
  --output_dir="${OUTPUT_DIR}" \
  --dataset_txt_paths_list "${TRAIN_TXT}","${FFHQ_TXT}" \
  --dataset_prob_paths_list 1,1 \
  --neg_prompt="painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth" \
  --lambda_lpips=2 \
  --lambda_l2=1 \
  --lambda_fde_wavelet=0.75 \
  --lambda_fde_residual=0.5 \
  --lambda_fde_high=0.15 \
  --fde_hidden_dim=128 \
  --fde_prompt_hidden_dim=256 \
  --fde_num_blocks=6 \
  --fde_residual_scale=0.25 \
  --deg_file_path="dataloaders/params_realesrgan.yml" \
  --tracker_project_name "train_refwrd_stage2"

