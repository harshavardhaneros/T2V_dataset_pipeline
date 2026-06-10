#!/usr/bin/env python3
"""Re-export empty/broken clip MP4s and refresh export/ symlinks for HTML review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.clip_io import export_clip_mp4  # noqa: E402
from common.review_clips import link_workspace_clips  # noqa: E402
from common.video_files import find_movie_video  # noqa: E402


def load_config(ws: Path) -> dict:
    import yaml

    cfg_path = _ROOT / "configs" / "pipeline_v3_2gpu.yaml"
    if not cfg_path.exists():
        cfg_path = _ROOT / "configs" / "pipeline_v3.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="Repair clip MP4s for HTML review")
    p.add_argument("--workspace", required=True)
    p.add_argument("--regenerate-html", action="store_true")
    args = p.parse_args()

    ws = Path(args.workspace).resolve()
    meta = ws / "metadata.jsonl"
    if not meta.exists():
        raise SystemExit(f"Missing {meta}")

    movie = find_movie_video(ws)
    if not movie:
        raise SystemExit(f"No source video in {ws}")

    config = load_config(ws)
    export_cfg = {**config.get("export", {}), "remove_watermark": False}
    thresholds = config.get("thresholds", {})
    clips_dir = ws / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(line) for line in meta.read_text().splitlines() if line.strip()]
    repaired = skipped = failed = 0

    for rec in records:
        clip_path = clips_dir / f"{rec['clip_id']}.mp4"
        if clip_path.exists():
            try:
                if clip_path.stat().st_size > 0:
                    skipped += 1
                    continue
            except OSError:
                pass
            clip_path.unlink(missing_ok=True)

        ok = export_clip_mp4(
            movie,
            rec,
            clip_path,
            export_cfg=export_cfg,
            thresholds=thresholds,
        )
        if ok:
            repaired += 1
        else:
            failed += 1
            print(f"FAIL {rec['clip_id']}", flush=True)

    export_dir = ws / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    clip_ids = [r["clip_id"] for r in records]
    link_workspace_clips(export_dir, ws, clip_ids)

    print(f"repaired={repaired} skipped={skipped} failed={failed}")

    if args.regenerate_html:
        import subprocess

        subprocess.run(
            [sys.executable, str(_ROOT / "scripts" / "view_workspace.py"), "--workspace", str(ws)],
            check=True,
        )


if __name__ == "__main__":
    main()
