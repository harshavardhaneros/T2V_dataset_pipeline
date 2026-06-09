#!/usr/bin/env bash
# Extract a time range from a movie for pipeline test runs.
# Example: minutes 100–130 (30 min) from ABCD.mp4
#
# Usage:
#   bash scripts/make_test_segment.sh \
#     /mnt/data0/harsha/Movies/feb_11/ABCD.mp4 \
#     100 130
#
# Output:
#   test_segments/ABCD_min100-130.mp4

set -euo pipefail
SRC="${1:?source video}"
START_MIN="${2:?start minute}"
END_MIN="${3:?end minute}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT}/test_segments"
STEM="$(basename "${SRC}" .mp4)"
OUT="${OUT_DIR}/${STEM}_min${START_MIN}-${END_MIN}.mp4"

mkdir -p "${OUT_DIR}"
if [[ -f "${OUT}" ]]; then
  echo "Already exists: ${OUT}"
  exit 0
fi

START_SEC=$((START_MIN * 60))
DUR_SEC=$(((END_MIN - START_MIN) * 60))

echo "Extracting ${START_MIN}:00 -> ${END_MIN}:00 (${DUR_SEC}s) from ${SRC}"
echo "Output: ${OUT}"

ffmpeg -hide_banner -y \
  -ss "${START_SEC}" \
  -i "${SRC}" \
  -t "${DUR_SEC}" \
  -c copy \
  -avoid_negative_ts make_zero \
  "${OUT}"

ls -lh "${OUT}"
echo "TIME_OFFSET_SEC=${START_SEC}"
