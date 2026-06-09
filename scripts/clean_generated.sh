#!/usr/bin/env bash
# Remove generated artifacts from code tree and reset external outputs.
set -euo pipefail
CODE="/mnt/data0/harsha/new_dataset_pipeline/Indic_video_pipeline"
OUT="/mnt/data0/harsha/new_dataset_pipeline/pipeline_outputs"

echo "Cleaning code tree artifacts..."
rm -rf "${CODE}/workspaces" "${CODE}/logs" "${CODE}/reports" "${CODE}/test_segments"
rm -rf "${CODE}/models"
find "${CODE}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Resetting outputs root..."
rm -rf "${OUT}"
mkdir -p "${OUT}/logs" "${OUT}/reports" "${OUT}/workspaces"

echo "Done. Code: ${CODE}  Outputs: ${OUT}"
