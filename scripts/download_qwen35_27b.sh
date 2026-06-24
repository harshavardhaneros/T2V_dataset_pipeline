#!/usr/bin/env bash
# Qwen3.5-27B for native MP4 video captioning (~54GB bf16).
set -euo pipefail
MODELS_ROOT="${MODELS_ROOT:-/mnt/data0/harsha/new_dataset_pipeline/models}"
MODEL_DIR="${MODELS_ROOT}/Qwen3.5-27B"
REPO="Qwen/Qwen3.5-27B"

mkdir -p "${MODELS_ROOT}"
if [[ -f "${MODEL_DIR}/config.json" ]]; then
  echo "Model already present: ${MODEL_DIR}"
  exit 0
fi

echo "Downloading ${REPO} -> ${MODEL_DIR}"
hf download "${REPO}" --local-dir "${MODEL_DIR}"
echo "Done: ${MODEL_DIR}"
