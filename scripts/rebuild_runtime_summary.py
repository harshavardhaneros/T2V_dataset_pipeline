#!/usr/bin/env python3
"""Rebuild runtime_summary.csv and pipeline_runtime.json from per-service logs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.base_service import ensure_path_on_syspath
from common.paths import workspaces_dir
from common.runtime_tracker import rebuild_runtime_artifacts
from common.video_files import find_movie_video
from run_pipeline import load_config, pipeline_root


def main() -> int:
    root = pipeline_root()
    ensure_path_on_syspath(root)
    parser = argparse.ArgumentParser(
        description="Rebuild runtime CSV/JSON from logs/s*/{video_id}_runtime.json"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline_v3_vllm_2gpu.yaml",
        help="Pipeline config under configs/",
    )
    parser.add_argument("--video-id", type=str, required=True, help="Workspace folder name")
    parser.add_argument(
        "--outputs-root",
        type=str,
        default=None,
        help="Override pipeline outputs root",
    )
    args = parser.parse_args()

    config = load_config(root, args.config)
    if args.outputs_root:
        config["pipeline"]["outputs_root"] = args.outputs_root

    movie_dir = workspaces_dir(config) / args.video_id
    if not movie_dir.exists():
        print(f"Workspace not found: {movie_dir}", file=sys.stderr)
        return 1

    movie_video = find_movie_video(movie_dir)
    movie_name = movie_video.name if movie_video else f"{args.video_id}.mp4"
    timings = rebuild_runtime_artifacts(
        config,
        video_id=args.video_id,
        movie_name=movie_name,
        movie_dir=movie_dir,
    )
    total = sum(timings.get(f"s{i}", 0) for i in range(1, 13))
    print(f"Rebuilt runtime for {args.video_id}: total={total:.2f}s")
    for step_id in sorted(timings):
        print(f"  {step_id}: {timings[step_id]:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
