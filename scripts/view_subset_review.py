#!/usr/bin/env python3
"""HTML review for the first N clips (captions, actors, buckets)."""

from __future__ import annotations

import argparse
import html
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
for p in (_ROOT, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from view_bucket_review import BUCKET_LABELS, write_bucket_html  # noqa: E402
from view_workspace import load_metadata, link_frames_for_review, write_all_clips_html  # noqa: E402
from common.review_clips import link_workspace_clips  # noqa: E402


def write_subset_index(
    export_dir: Path,
    *,
    prefix: str,
    n: int,
    records: list[dict],
) -> None:
    tagged = sum(1 for r in records if r.get("actor_status") == "tagged")
    captioned = sum(1 for r in records if (r.get("caption") or "").strip())
    buckets = Counter(r.get("bucket", "?") for r in records)
    bucket_lines = "".join(
        f"<li>{html.escape(BUCKET_LABELS.get(b, b))}: {c}</li>"
        for b, c in sorted(buckets.items())
    )
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>First {n} clips — review</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #111; color: #eee;
    max-width: 720px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
  a.card {{
    display: block; padding: 1rem 1.25rem; margin: 0.5rem 0;
    background: #1e2a3a; border: 1px solid #3a5a7a; border-radius: 8px;
    color: #cde; text-decoration: none; font-weight: 500;
  }}
  a.card:hover {{ background: #2a3a4a; }}
  a.card small {{ display: block; color: #89a; font-weight: 400; margin-top: 0.25rem; }}
  .stats {{ color: #aaa; }}
</style></head><body>
<h1>First {n} clips</h1>
<p class="stats">{tagged} actor-tagged · {captioned} captioned · {len(buckets)} buckets</p>
<ul>{bucket_lines}</ul>
<p><a href="index.html">← full review hub</a></p>
<a class="card" href="{prefix}_all_clips_review.html">All clips (grid)
  <small>Video + captions + actors + scores · filterable</small></a>
<a class="card" href="{prefix}_bucket_review.html">Bucket review
  <small>Grouped by bucket · 2×2 paginated</small></a>
<a class="card" href="actor_caption_report.html">Actor vs caption (full dataset)
  <small>All tagged clips in workspace</small></a>
</body></html>"""
    out = export_dir / f"{prefix}_index.html"
    out.write_text(page, encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build HTML review for first N clips")
    p.add_argument("--workspace", required=True)
    p.add_argument("--max-clips", type=int, default=35)
    p.add_argument("--prefix", default="first_35")
    p.add_argument("--grid-cols", type=int, default=2)
    p.add_argument("--grid-rows", type=int, default=2)
    args = p.parse_args()

    ws = Path(args.workspace).resolve()
    export_dir = ws / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    records = load_metadata(ws)[: args.max_clips]
    if not records:
        raise SystemExit("No records in metadata.jsonl")

    clip_ids = [r["clip_id"] for r in records]
    link_workspace_clips(export_dir, ws, clip_ids)
    link_frames_for_review(export_dir, ws, clip_ids)

    all_out = export_dir / f"{args.prefix}_all_clips_review.html"
    write_all_clips_html(
        records,
        export_dir,
        all_out,
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
    )

    buckets: dict[str, list[dict]] = {}
    for rec in records:
        buckets.setdefault(rec.get("bucket", "unknown"), []).append(rec)

    bucket_out = export_dir / f"{args.prefix}_bucket_review.html"
    write_bucket_html(
        buckets,
        export_dir,
        bucket_out,
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
    )

    write_subset_index(export_dir, prefix=args.prefix, n=len(records), records=records)

    print(f"\nOpen: http://localhost:8888/{args.prefix}_index.html")
    print(f"  or: http://localhost:8888/{args.prefix}_all_clips_review.html")
    print(f"  or: http://localhost:8888/{args.prefix}_bucket_review.html")


if __name__ == "__main__":
    main()
