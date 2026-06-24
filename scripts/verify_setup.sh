#!/usr/bin/env bash
# Verify GPU + deps + Qwen model before full pipeline run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${ROOT}/../models/Qwen2.5-VL-7B-Instruct"
YOLO="${ROOT}/master/actors/yolov12n-face.pt"

# Use activated env python (conda run can fail on broken base cross-compiler toolchains).
if [[ "${CONDA_DEFAULT_ENV:-}" != "indic_video_pipeline" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate indic_video_pipeline
fi

echo "=== Python / CUDA (indic_video_pipeline) ==="
python <<PY
import sys
errors = 0

def check(cond, msg):
    global errors
    if cond:
        print(f"  OK  {msg}")
    else:
        print(f"  FAIL {msg}")
        errors += 1

try:
    import torch
    check(torch.cuda.is_available(), f"torch {torch.__version__} — {torch.cuda.device_count()} GPUs")
except Exception as e:
    check(False, f"torch: {e}")

for pkg in ("ray", "ultralytics", "insightface", "scenedetect", "transformers", "qwen_vl_utils"):
    try:
        __import__(pkg)
        check(True, pkg)
    except ImportError as e:
        check(False, f"{pkg}: {e}")

import os
model = "${MODEL}"
cfg = os.path.join(model, "config.json")
check(os.path.isfile(cfg), f"Qwen-7B config at {model}")
if os.path.isdir(model):
    weights = [f for f in os.listdir(model) if f.endswith((".safetensors", ".bin"))]
    check(len(weights) > 0, f"Qwen-7B weights ({len(weights)} shard files)")

sys.exit(errors)
PY

echo ""
echo "=== Actor tagging assets ==="
if [[ -f "${YOLO}" ]]; then
  echo "  OK  YOLO: ${YOLO}"
else
  echo "  FAIL Run: bash scripts/download_yolo_face.sh"
fi
emb_count=$(ls "${ROOT}/master/actors/actor_embeddings/"*.pkl 2>/dev/null | wc -l)
echo "  OK  Actor embeddings: ${emb_count} pkl files"

echo ""
echo "=== pipeline.yaml ==="
grep -E "model_path|actor_tag_gpu|caption_gpu|caption_backend" "${ROOT}/configs/pipeline.yaml"

echo ""
echo "=== Ready command ==="
echo "  conda activate indic_video_pipeline"
echo "  cd ${ROOT}"
echo "  python run_pipeline.py --movie /mnt/data0/harsha/Movies/feb_11/ABCD.mp4 --video-id ABCD --force"
