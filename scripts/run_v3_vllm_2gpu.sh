#!/usr/bin/env bash
# v3 pipeline with vLLM batched s5 + s8 on 2 GPUs (tensor parallel).
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MOVIE="${1:-/mnt/data0/harsha/Movies/feb_11/Devdas_20min_to_50min.mp4}"
VIDEO_ID="${2:-Devdas_vllm_test}"
GPUS="${CUDA_VISIBLE_DEVICES:-4,5}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate indic_video_pipeline

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

# Gemma4 needs vLLM 0.19+ (V1 engine). Qwen2.5 on 0.8.x uses legacy env — set only for old vLLM.
VLLM_VER="$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo 0)"
VLLM_MAJOR="${VLLM_VER%%.*}"
VLLM_MINOR="$(echo "${VLLM_VER}" | cut -d. -f2)"
if [[ "${VLLM_MAJOR}" -eq 0 && "${VLLM_MINOR}" -lt 19 ]]; then
  export VLLM_USE_V1=0
  export VLLM_ATTENTION_BACKEND=XFORMERS
fi

python -c "import vllm; print('vllm', vllm.__version__)" || {
  echo "Run: bash scripts/install_vllm.sh (qwen2.5) or bash scripts/install_vllm_gemma4.sh (gemma4)"
  exit 1
}

echo "=== v3 vLLM 2-GPU pipeline ==="
echo "  movie:    ${MOVIE}"
echo "  video-id: ${VIDEO_ID}"
echo "  GPUs:     ${CUDA_VISIBLE_DEVICES}"
echo "  config:   pipeline_v3_vllm_2gpu.yaml"
echo ""

# Two phases avoid Ray (s6/s7) breaking the second vLLM load (s8) in one process.
python run_pipeline.py \
  --config pipeline_v3_vllm_2gpu.yaml \
  --movie "${MOVIE}" \
  --video-id "${VIDEO_ID}" \
  --from-step s1 --to-step s7 \
  --force

python run_pipeline.py \
  --config pipeline_v3_vllm_2gpu.yaml \
  --movie "${MOVIE}" \
  --video-id "${VIDEO_ID}" \
  --from-step s8 --to-step s12 \
  --force

echo ""
echo "Outputs: /mnt/data0/harsha/new_dataset_pipeline/v3_outputs/${VIDEO_ID}/"
