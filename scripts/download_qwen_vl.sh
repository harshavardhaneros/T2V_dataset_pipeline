#!/usr/bin/env bash
# Download Qwen2.5-VL-32B-Instruct into pipeline-local models/ folder.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${ROOT}/models/Qwen2.5-VL-32B-Instruct"
REPO="Qwen/Qwen2.5-VL-32B-Instruct"

mkdir -p "$(dirname "$MODEL_DIR")"
if [[ -f "${MODEL_DIR}/config.json" ]]; then
  echo "Model already present: ${MODEL_DIR}"
  exit 0
fi

echo "Downloading ${REPO} -> ${MODEL_DIR}"
echo "This is ~60GB+; ensure enough disk space on /mnt/data0"

conda run -n indic_video_pipeline hf download "${REPO}" \
  --local-dir "${MODEL_DIR}"

echo "Done: ${MODEL_DIR}"
ls -la "${MODEL_DIR}" | head
