#!/usr/bin/env bash
# Qwen3-VL-32B-Instruct for vLLM captioning (~67GB). Needs multi-GPU TP (e.g. 4–8× H100).
set -euo pipefail
MODELS_ROOT="${MODELS_ROOT:-/mnt/data0/harsha/new_dataset_pipeline/models}"
MODEL_DIR="${MODELS_ROOT}/Qwen3-VL-32B-Instruct"
REPO="Qwen/Qwen3-VL-32B-Instruct"

mkdir -p "${MODELS_ROOT}"
if [[ -f "${MODEL_DIR}/config.json" ]]; then
  echo "Model already present: ${MODEL_DIR}"
  exit 0
fi

echo "Downloading ${REPO} -> ${MODEL_DIR}"
echo "This is ~67GB; ensure enough disk space on ${MODELS_ROOT}"

hf download "${REPO}" --local-dir "${MODEL_DIR}"
echo "Done: ${MODEL_DIR}"
