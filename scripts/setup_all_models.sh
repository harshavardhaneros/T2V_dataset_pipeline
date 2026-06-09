#!/usr/bin/env bash
# Download / relocate models into shared models_root.
set -euo pipefail
MODELS="/mnt/data0/harsha/new_dataset_pipeline/models"
PIPE_MODELS="/mnt/data0/harsha/new_dataset_pipeline/Indic_video_pipeline/models"
MASTER="/mnt/data0/harsha/new_dataset_pipeline/Master_Pipeline_t2i_dataset"
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

# YOLO face
YOLO="${MODELS}/yolov12n-face.pt"
if [[ ! -f "${YOLO}" ]]; then
  if [[ -f "${MASTER}/actors/yolov12n-face.pt" ]]; then
    cp "${MASTER}/actors/yolov12n-face.pt" "${YOLO}"
  else
    URL="https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12n-face.pt"
    curl -L -o "${YOLO}" "${URL}"
  fi
fi

echo "Models ready:"
ls -lh "${MODELS}/Qwen2.5-VL-32B-Instruct/config.json" "${YOLO}"
echo "Actor embeddings (Master, not duplicated): ${MASTER}/actors/actor_embeddings/*.pkl"
