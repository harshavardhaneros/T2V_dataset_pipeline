#!/usr/bin/env bash
# Download YOLO face detector for actor tagging (5.3MB)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)/Master_Pipeline_t2i_dataset"
OUT="${ROOT}/actors/yolov12n-face.pt"
mkdir -p "$(dirname "$OUT")"
if [[ -f "$OUT" ]]; then
  echo "Already exists: $OUT"
  exit 0
fi
URL="https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov12n-face.pt"
echo "Downloading $URL -> $OUT"
curl -L -o "$OUT" "$URL"
ls -la "$OUT"
