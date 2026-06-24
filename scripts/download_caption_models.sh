#!/usr/bin/env bash
# Download caption model weights into shared models/ directory.
# Usage:
#   bash scripts/download_caption_models.sh qwen3 gemma4
#   bash scripts/download_caption_models.sh all
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MODELS_ROOT="${MODELS_ROOT:-/mnt/data0/harsha/new_dataset_pipeline/models}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {qwen2.5|qwen3|qwen3.5|gemma3|gemma4|gemma4_dense|all} ..."
  exit 1
fi

download_one() {
  case "$1" in
    qwen2.5|qwen25) bash "${ROOT}/scripts/download_qwen_vl_7b.sh" ;;
    qwen3)          bash "${ROOT}/scripts/download_qwen3_vl_32b.sh" ;;
    qwen3.5|qwen35) bash "${ROOT}/scripts/download_qwen35_27b.sh" ;;
    gemma3)         hf download google/gemma-3-4b-it --local-dir "${MODELS_ROOT}/gemma-3-4b-it" ;;
    gemma4)         bash "${ROOT}/scripts/download_gemma4_31b.sh" ;;
    gemma4_dense)   bash "${ROOT}/scripts/download_gemma4_31b_dense.sh" ;;
    all)
      download_one qwen2.5
      download_one qwen3
      download_one qwen3.5
      download_one gemma3
      download_one gemma4
      download_one gemma4_dense
      ;;
    *) echo "Unknown model: $1"; exit 1 ;;
  esac
}

mkdir -p "${MODELS_ROOT}"
for arg in "$@"; do
  download_one "$arg"
done
