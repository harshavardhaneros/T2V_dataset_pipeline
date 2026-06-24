#!/usr/bin/env bash
# v3 pipeline with gemma4_dense native MP4 caption (HF) + vLLM s4/s5 on 2 GPUs.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=4,5 bash scripts/run_v3_dense_video_2gpu.sh /path/to/movies_folder/
#   CUDA_VISIBLE_DEVICES=4,5 bash scripts/run_v3_dense_video_2gpu.sh /path/to/movie.mp4 [VIDEO_ID]
#
# Outputs: v3_outputs_dense/<VIDEO_ID>/  (separate from v3_outputs/ vLLM caption runs)
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="pipeline_v3_dense_video_2gpu.yaml"
export CONFIG

exec bash "${ROOT}/scripts/run_v3_vllm_2gpu.sh" "$@"
