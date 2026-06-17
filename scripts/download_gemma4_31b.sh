#!/usr/bin/env bash
# Gemma-4-31B-IT for transformers captioning (~62GB bf16). Fits 1×80GB or multi-GPU auto.
set -euo pipefail
MODELS_ROOT="${MODELS_ROOT:-/mnt/data0/harsha/new_dataset_pipeline/models}"
MODEL_DIR="${MODELS_ROOT}/gemma-4-31b-it"
REPO="google/gemma-4-31B-it"

mkdir -p "${MODELS_ROOT}"
if [[ -f "${MODEL_DIR}/config.json" ]]; then
  echo "Model already present: ${MODEL_DIR}"
  exit 0
fi

echo "Downloading ${REPO} -> ${MODEL_DIR}"
echo "Gemma models may require: huggingface-cli login (accept license on HF)"

hf download "${REPO}" --local-dir "${MODEL_DIR}"
echo "Done: ${MODEL_DIR}"
