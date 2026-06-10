#!/usr/bin/env bash
# v3 pipeline with vLLM batched s5 + s8 on 8 GPUs (2×TP=4 replicas) + Ray s6/s7/s9.
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MOVIE="${1:-/mnt/data0/harsha/Movies/feb_11/Devdas_20min_to_50min.mp4}"
VIDEO_ID="${2:-Devdas_vllm_8gpu_test}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate indic_video_pipeline

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS

python -c "import vllm; print('vllm', vllm.__version__)" || {
  echo "Run first: bash scripts/install_vllm.sh"
  exit 1
}

python -c "import ray; print('ray', ray.__version__)" || {
  echo "Ray missing — run: pip install 'ray>=2.9.0'"
  exit 1
}

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
if [[ "${NGPU}" -lt 8 ]]; then
  echo "WARNING: ${NGPU} GPU(s) visible (expected 8). Check CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

echo "=== v3 vLLM 8-GPU pipeline ==="
echo "  movie:    ${MOVIE}"
echo "  video-id: ${VIDEO_ID}"
echo "  GPUs:     ${CUDA_VISIBLE_DEVICES} (${NGPU} visible)"
echo "  config:   pipeline_v3_vllm_8gpu.yaml"
echo "  s5/s8:    vLLM 2×TP=4 replicas, batch=24"
echo "  s6/s7/s9: Ray parallel (8 workers)"
echo ""

python run_pipeline.py \
  --config pipeline_v3_vllm_8gpu.yaml \
  --movie "${MOVIE}" \
  --video-id "${VIDEO_ID}" \
  --force

echo ""
echo "Outputs: /mnt/data0/harsha/new_dataset_pipeline/v3_outputs/${VIDEO_ID}/"
echo "HTML review:"
echo "  python scripts/view_workspace.py --workspace /mnt/data0/harsha/new_dataset_pipeline/v3_outputs/${VIDEO_ID}"
