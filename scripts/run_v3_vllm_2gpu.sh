#!/usr/bin/env bash
# v3 pipeline with vLLM batched s4/s5 + s8 on 2 GPUs (data-parallel, TP=1 per GPU).
#
# Usage (single movie):
#   CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_v3_vllm_2gpu.sh /path/to/movie.mp4 [VIDEO_ID]
#
# Usage (all movies in a folder):
#   CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_v3_vllm_2gpu.sh /path/to/movies_folder/
#
# VIDEO_ID defaults to the movie filename stem (e.g. Devdas.mp4 → Devdas).
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INPUT="${1:-/mnt/data0/harsha/Movies/feb_11/Devdas_20min_to_50min.mp4}"
VIDEO_ID_OVERRIDE="${2:-}"
CONFIG="${CONFIG:-pipeline_v3_vllm_2gpu.yaml}"
GPUS="${CUDA_VISIBLE_DEVICES:-4,5}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate indic_video_pipeline

cd "${ROOT}"
mkdir -p "${ROOT}/logs"
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

# shellcheck source=scripts/run_v3_vllm_phases.sh
source "${ROOT}/scripts/run_v3_vllm_phases.sh"

echo "=== v3 vLLM 2-GPU pipeline ==="
echo "  input:  ${INPUT}"
echo "  GPUs:   ${CUDA_VISIBLE_DEVICES}"
echo "  config: ${CONFIG}"
echo ""

FAILED=()
SUCCEEDED=()

run_batch_dir() {
  local DIR="$1"
  mapfile -t MOVIE_LINES < <(python - "${DIR}" <<'PY'
import sys
from pathlib import Path

from common.video_files import list_movie_videos

directory = Path(sys.argv[1])
movies = list_movie_videos(directory)
if not movies:
    raise SystemExit(f"No video files found in {directory}")
for movie in movies:
    print(f"{movie}\t{movie.stem}")
PY
)

  local TOTAL="${#MOVIE_LINES[@]}"
  echo "Found ${TOTAL} movie(s) in ${DIR}"
  echo ""

  local IDX=0
  for line in "${MOVIE_LINES[@]}"; do
    IDX=$((IDX + 1))
    local MOVIE="${line%%$'\t'*}"
    local VID="${line#*$'\t'}"
    echo "────────────────────────────────────────"
    echo "[${IDX}/${TOTAL}] ${VID}"
    echo "────────────────────────────────────────"
    if run_v3_vllm_phases "${MOVIE}" "${VID}" "${CONFIG}"; then
      SUCCEEDED+=("${VID}")
    else
      FAILED+=("${VID}")
      echo "ERROR: pipeline failed for ${VID} (${MOVIE})" >&2
    fi
  done
}

if [[ -d "${INPUT}" ]]; then
  if [[ -n "${VIDEO_ID_OVERRIDE}" ]]; then
    echo "WARNING: VIDEO_ID override ignored in folder mode (each movie uses its filename stem)" >&2
  fi
  run_batch_dir "${INPUT}"
elif [[ -f "${INPUT}" ]]; then
  VID="${VIDEO_ID_OVERRIDE:-$(python - "${INPUT}" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).stem)
PY
)}"
  if run_v3_vllm_phases "${INPUT}" "${VID}" "${CONFIG}"; then
    SUCCEEDED+=("${VID}")
  else
    FAILED+=("${VID}")
    exit 1
  fi
else
  echo "ERROR: not a file or directory: ${INPUT}" >&2
  echo "Usage: bash scripts/run_v3_vllm_2gpu.sh /path/to/movie.mp4 [VIDEO_ID]" >&2
  echo "   or: bash scripts/run_v3_vllm_2gpu.sh /path/to/movies_folder/" >&2
  exit 1
fi

echo ""
echo "========== batch summary =========="
echo "  succeeded: ${#SUCCEEDED[@]} — ${SUCCEEDED[*]:-none}"
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "  failed:    ${#FAILED[@]} — ${FAILED[*]}"
  exit 1
fi
