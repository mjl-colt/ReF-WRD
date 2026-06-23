#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: bash scripts/reproducibility/run_test_refwrd.sh INPUT_LQ_DIR OUTPUT_DIR"
  exit 1
fi

: "${SD21BASE_PATH:?Set SD21BASE_PATH to the Stable Diffusion 2.1 base directory}"
: "${RAM_PATH:?Set RAM_PATH to the RAM checkpoint}"
: "${DAPE_PATH:?Set DAPE_PATH to the DAPE/RAM fine-tuned checkpoint}"
: "${REFWRD_CKPT:?Set REFWRD_CKPT to the ReF-WRD Stage-II checkpoint}"

INPUT_DIR="$1"
OUTPUT_DIR="$2"
FDE_RESIDUAL_SCALE="${FDE_RESIDUAL_SCALE:-0.25}"

python test_fde_sr.py \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_DIR}" \
  --fde_sr_path "${REFWRD_CKPT}" \
  --pretrained_model_name_or_path "${SD21BASE_PATH}" \
  --ram_ft_path "${DAPE_PATH}" \
  --ram_path "${RAM_PATH}" \
  --fde_residual_scale "${FDE_RESIDUAL_SCALE}" \
  --save_intermediate

echo "Used fixed residual scale --fde_residual_scale=${FDE_RESIDUAL_SCALE}."
