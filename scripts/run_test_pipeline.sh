#!/usr/bin/env bash
# Test pipeline on ABCD minutes 100–130 (~30 min segment).
# Phase A: s1–s7 (~5–10 min). Phase B: s8 with 15 real Qwen captions (~10–20 min).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="/mnt/data0/harsha/Movies/feb_11/ABCD.mp4"
VID="ABCD_test_100_130"
SEG="${ROOT}/test_segments/ABCD_min100-130.mp4"
OFFSET_SEC=6000   # 100 * 60

cd "${ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate indic_video_pipeline

echo "=== 1) Extract 30-min segment (min 100–130) ==="
bash scripts/make_test_segment.sh "${SRC}" 100 130

echo "=== 2) Phase A: extract → actor tagging (stop at s7) ==="
python run_pipeline.py \
  --movie "${SEG}" \
  --video-id "${VID}" \
  --time-offset "${OFFSET_SEC}" \
  --force \
  --to-step s7

echo "=== 3) Inspect Phase A outputs ==="
python scripts/inspect_test_output.py --workspace "workspaces/${VID}"

echo "=== 4) Phase B: caption test (15 clips with real Qwen) ==="
python run_pipeline.py \
  --movie "${SEG}" \
  --video-id "${VID}" \
  --time-offset "${OFFSET_SEC}" \
  --from-step s8 \
  --to-step s12 \
  --max-clips 15 \
  --force

echo "=== 5) Final inspect ==="
python scripts/inspect_test_output.py --workspace "workspaces/${VID}" --show-captions

echo "Done. Workspace: workspaces/${VID}"
