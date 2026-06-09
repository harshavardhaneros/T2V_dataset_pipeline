#!/usr/bin/env bash
# Fast native-video captioning model (~15GB). Recommended over 32B for 5s clips.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_ROOT="${MODELS_ROOT:-/mnt/data0/harsha/new_dataset_pipeline/models}"
MODEL_DIR="${MODELS_ROOT}/Qwen2.5-VL-7B-Instruct"
REPO="Qwen/Qwen2.5-VL-7B-Instruct"

mkdir -p "${MODELS_ROOT}"
if [[ -f "${MODEL_DIR}/config.json" ]]; then
  echo "Model already present: ${MODEL_DIR}"
  exit 0
fi

echo "Downloading ${REPO} -> ${MODEL_DIR}"
hf download "${REPO}" --local-dir "${MODEL_DIR}"
echo "Done: ${MODEL_DIR}"
