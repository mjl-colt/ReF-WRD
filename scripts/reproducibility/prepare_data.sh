#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: bash scripts/reproducibility/prepare_data.sh LSDIR_HR_DIR FFHQ_DIR OUTPUT_LIST_DIR"
  exit 1
fi

LSDIR_HR_DIR="$1"
FFHQ_DIR="$2"
OUTPUT_LIST_DIR="$3"

mkdir -p "${OUTPUT_LIST_DIR}"

find "${LSDIR_HR_DIR}" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.webp' \) \
  | sort > "${OUTPUT_LIST_DIR}/lsdir_train.txt"

find "${FFHQ_DIR}" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' -o -iname '*.webp' \) \
  | sort | head -n 10000 > "${OUTPUT_LIST_DIR}/ffhq_first10k.txt"

echo "Wrote ${OUTPUT_LIST_DIR}/lsdir_train.txt"
echo "Wrote ${OUTPUT_LIST_DIR}/ffhq_first10k.txt"

