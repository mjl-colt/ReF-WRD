#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: bash scripts/reproducibility/run_metrics.sh RESTORED_DIR GT_DIR LOG_DIR"
  exit 1
fi

RESTORED_DIR="$1"
GT_DIR="$2"
LOG_DIR="$3"

python test_metrics.py \
  --inp_imgs "${RESTORED_DIR}" \
  --gt_imgs "${GT_DIR}" \
  --log "${LOG_DIR}"

