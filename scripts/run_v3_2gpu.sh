#!/usr/bin/env bash
# Full v3 pipeline on 2 GPUs (Ray parallel s5/s7/s8, DOVER on cuda:1).
# Usage:
#   bash scripts/run_v3_2gpu.sh
#   bash scripts/run_v3_2gpu.sh /path/to/movie.mp4 MY_VIDEO_ID
#   CUDA_VISIBLE_DEVICES=4,5 bash scripts/run_v3_2gpu.sh ...
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MOVIE="${1:-/mnt/data0/harsha/Movies/feb_11/Devdas_20min_to_50min.mp4}"
VIDEO_ID="${2:-Devdas_20min_to_50min1}"
GPUS="${CUDA_VISIBLE_DEVICES:-4,5}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate indic_video_pipeline

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

python -c "import ray; print('ray', ray.__version__)" || {
  echo "Ray missing — run: pip install 'ray>=2.9.0'"
  exit 1
}

echo "=== v3 2-GPU pipeline ==="
echo "  movie:    ${MOVIE}"
echo "  video-id: ${VIDEO_ID}"
echo "  GPUs:     ${CUDA_VISIBLE_DEVICES}"
echo "  config:   pipeline_v3_2gpu.yaml"
echo ""

python run_pipeline.py \
  --config pipeline_v3_2gpu.yaml \
  --movie "${MOVIE}" \
  --video-id "${VIDEO_ID}" \
  --force

echo ""
echo "Done. Outputs: /mnt/data0/harsha/new_dataset_pipeline/v3_outputs/${VIDEO_ID}/"
