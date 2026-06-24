#!/usr/bin/env bash
# Download / relocate models into shared models_root.
set -euo pipefail
MODELS="/mnt/data0/harsha/new_dataset_pipeline/models"
PIPE_MODELS="/mnt/data0/harsha/new_dataset_pipeline/Indic_video_pipeline/models"
ROOT="/mnt/data0/harsha/new_dataset_pipeline/Indic_video_pipeline"

mkdir -p "${MODELS}"

# Move Qwen from pipeline tree if present
if [[ -d "${PIPE_MODELS}/Qwen2.5-VL-32B-Instruct" ]] && [[ ! -f "${MODELS}/Qwen2.5-VL-32B-Instruct/config.json" ]]; then
  echo "Moving Qwen2.5-VL to ${MODELS}..."
  mv "${PIPE_MODELS}/Qwen2.5-VL-32B-Instruct" "${MODELS}/"
fi

# Download Qwen if missing
if [[ ! -f "${MODELS}/Qwen2.5-VL-32B-Instruct/config.json" ]]; then
  echo "Downloading Qwen2.5-VL-32B-Instruct (~64GB)..."
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate indic_video_pipeline
  hf download Qwen/Qwen2.5-VL-32B-Instruct --local-dir "${MODELS}/Qwen2.5-VL-32B-Instruct"
fi

# YOLO face (shared models + master/actors for s7)
YOLO="${MODELS}/yolov12n-face.pt"
MASTER_YOLO="${ROOT}/master/actors/yolov12n-face.pt"
if [[ ! -f "${YOLO}" ]]; then
  URL="https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12n-face.pt"
  curl -L -o "${YOLO}" "${URL}"
fi
mkdir -p "${ROOT}/master/actors"
if [[ ! -f "${MASTER_YOLO}" ]]; then
  ln -sfn "${YOLO}" "${MASTER_YOLO}" 2>/dev/null || cp "${YOLO}" "${MASTER_YOLO}"
fi

echo "Models ready:"
ls -lh "${MODELS}/Qwen2.5-VL-32B-Instruct/config.json" "${YOLO}" 2>/dev/null || true
echo "Actor embeddings: ${ROOT}/master/actors/actor_embeddings/*.pkl"
