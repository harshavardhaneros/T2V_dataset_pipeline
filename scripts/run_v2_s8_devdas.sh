#!/usr/bin/env bash
# Re-caption Devdas with actor names into v2_outputs (copies s1–s7 workspace first).
set -euo pipefail

ROOT="/mnt/data0/harsha/new_dataset_pipeline"
PIPE="$ROOT/Indic_video_pipeline"
VID="devdas_standard"
MOVIE="/mnt/data0/parth/world_models/HunyuanVideo-Avatar/assets/devdas_standard.mp4"

SRC="$ROOT/pipeline_outputs/workspaces/$VID"
DST="$ROOT/v2_outputs/workspaces/$VID"

mkdir -p "$ROOT/v2_outputs"/{workspaces,logs,reports}

if [[ ! -f "$SRC/metadata.jsonl" ]]; then
  echo "Missing $SRC/metadata.jsonl — run v1 pipeline first."
  exit 1
fi

mkdir -p "$DST"

echo "Seeding v2 workspace from $SRC ..."
rsync -a --delete \
  "$SRC/metadata.jsonl" \
  "$SRC/actor_frames/" \
  "$SRC/actor_tags/" \
  "$SRC/movie_watermark.json" \
  "$DST/"

# Symlink source video if not present
mkdir -p "$DST"
if [[ ! -e "$DST/$(basename "$MOVIE")" ]]; then
  ln -sf "$MOVIE" "$DST/$(basename "$MOVIE")"
fi

cd "$PIPE"
export PYTHONPATH="$PIPE${PYTHONPATH:+:$PYTHONPATH}"

python run_pipeline.py \
  --config configs/pipeline_v2.yaml \
  --movie "$MOVIE" \
  --video-id "$VID" \
  --from-step s8 \
  --to-step s12 \
  --force

echo "Done. Metadata: $DST/metadata.jsonl"
echo "Review: python scripts/view_samples.py --workspace $DST -n 8 --tagged-only"
