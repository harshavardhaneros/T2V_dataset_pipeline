#!/usr/bin/env bash
# Run v3 vLLM two-phase pipeline for one movie (s1-s7, then s8-s12).
# Sourced by run_v3_vllm_2gpu.sh or invoked directly.
set -eo pipefail

run_v3_vllm_phases() {
  local MOVIE="$1"
  local VIDEO_ID="$2"
  local CONFIG="${3:-pipeline_v3_vllm_2gpu.yaml}"
  local PHASE1_RC PHASE2_RC

  if [[ ! -f "${MOVIE}" ]]; then
    echo "ERROR: movie not found: ${MOVIE}" >&2
    return 1
  fi

  echo ""
  echo "=== v3 vLLM pipeline: ${VIDEO_ID} ==="
  echo "  movie:    ${MOVIE}"
  echo "  video-id: ${VIDEO_ID}"
  echo "  config:   ${CONFIG}"
  echo "  GPUs:     ${CUDA_VISIBLE_DEVICES}"
  echo ""

  set +e
  python run_pipeline.py \
    --config "${CONFIG}" \
    --movie "${MOVIE}" \
    --video-id "${VIDEO_ID}" \
    --from-step s1 --to-step s7 \
    --force
  PHASE1_RC=$?
  set -e

  if [[ "${PHASE1_RC}" -ne 0 ]]; then
    echo "ERROR: phase 1 (s1-s7) failed for ${VIDEO_ID} (exit ${PHASE1_RC}) — skipping phase 2" >&2
    return "${PHASE1_RC}"
  fi

  set +e
  python run_pipeline.py \
    --config "${CONFIG}" \
    --movie "${MOVIE}" \
    --video-id "${VIDEO_ID}" \
    --from-step s8 --to-step s12 \
    --force
  PHASE2_RC=$?
  set -e

  if [[ "${PHASE2_RC}" -ne 0 ]]; then
    echo "ERROR: phase 2 (s8-s12) failed for ${VIDEO_ID} (exit ${PHASE2_RC})" >&2
    return "${PHASE2_RC}"
  fi

  python scripts/rebuild_runtime_summary.py \
    --config "${CONFIG}" \
    --video-id "${VIDEO_ID}" || {
    echo "WARNING: rebuild_runtime_summary failed for ${VIDEO_ID}" >&2
  }

  local OUT_ROOT
  OUT_ROOT="$(python -c "
import yaml
from pathlib import Path
cfg = yaml.safe_load(open('configs/${CONFIG}'))
print(Path(cfg['outputs_root']) / '${VIDEO_ID}')
")"

  echo ""
  echo "Done: ${VIDEO_ID}"
  echo "  outputs: ${OUT_ROOT}/"
  echo "  runtime: ${OUT_ROOT}/reports/runtime_summary.csv"
}
