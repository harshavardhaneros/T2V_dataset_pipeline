#!/usr/bin/env bash
# Gemma-4-31B Dense (IT) for native MP4 video captioning.
set -euo pipefail
MODELS_ROOT="${MODELS_ROOT:-/mnt/data0/harsha/new_dataset_pipeline/models}"
MODEL_DIR="${MODELS_ROOT}/gemma-4-31b-dense"
REPO="google/gemma-4-31B-it"
SRC="${MODELS_ROOT}/gemma-4-31b-it"

mkdir -p "${MODELS_ROOT}"
if [[ -f "${MODEL_DIR}/config.json" ]]; then
  echo "Model already present: ${MODEL_DIR}"
  exit 0
fi

if [[ -f "${SRC}/config.json" ]]; then
  ln -sfn "${SRC}" "${MODEL_DIR}"
  echo "Linked ${MODEL_DIR} -> ${SRC}"
  exit 0
fi

echo "Downloading ${REPO} -> ${MODEL_DIR}"
hf download "${REPO}" --local-dir "${MODEL_DIR}"
echo "Done: ${MODEL_DIR}"
