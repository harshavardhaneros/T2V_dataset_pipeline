#!/usr/bin/env bash
# Full v3 pipeline on 8 GPUs — parallel s5/s6/s7/s8/s9 via Ray.
#
# Usage:
#   bash scripts/run_v3_8gpu.sh
#   bash scripts/run_v3_8gpu.sh /path/to/movie.mp4 MY_VIDEO_ID
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/run_v3_8gpu.sh ...
#
# Resume from step N (recommended after partial run):
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_pipeline.py \
#     --config pipeline_v3_8gpu.yaml --video-id MY_VIDEO_ID --from-step s8
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MOVIE="${1:-/mnt/data0/harsha/Movies/feb_11/Devdas_20min_to_50min.mp4}"
VIDEO_ID="${2:-Devdas_8gpu_test}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate indic_video_pipeline

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

python -c "import ray; print('ray', ray.__version__)" || {
  echo "Ray missing — run: pip install 'ray>=2.9.0'"
  exit 1
}

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
if [[ "${NGPU}" -lt 8 ]]; then
  echo "WARNING: ${NGPU} GPU(s) visible (expected 8). Check CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

echo "=== v3 8-GPU pipeline ==="
echo "  movie:    ${MOVIE}"
echo "  video-id: ${VIDEO_ID}"
echo "  GPUs:     ${CUDA_VISIBLE_DEVICES} (${NGPU} visible)"
echo "  config:   pipeline_v3_8gpu.yaml"
echo ""

python run_pipeline.py \
  --config pipeline_v3_8gpu.yaml \
  --movie "${MOVIE}" \
  --video-id "${VIDEO_ID}" \
  --force

echo ""
echo "Done. Outputs: /mnt/data0/harsha/new_dataset_pipeline/v3_outputs/${VIDEO_ID}/"
echo "HTML review:"
echo "  python scripts/view_workspace.py --workspace /mnt/data0/harsha/new_dataset_pipeline/v3_outputs/${VIDEO_ID}"
