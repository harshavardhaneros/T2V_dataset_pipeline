#!/usr/bin/env python3
"""HTML review grouped by bucket: video clips + captions (v3 export layout)."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.caption_text import caption_for_review, caption_to_str  # noqa: E402
from common.review_clips import clip_url, link_clip_under_export  # noqa: E402

BUCKET_LABELS = {
    "bucket_01": "People & portraits",
    "bucket_02": "Clothing & textiles",
    "bucket_03": "Architecture",
    "bucket_04": "Landscape & nature",
    "bucket_05": "Urban street",
    "bucket_06": "Rural village",
    "bucket_07": "Food & drink",
    "bucket_08": "Festivals & rituals",
    "bucket_09": "Objects & artifacts",
    "bucket_10": "Animals & wildlife",
    "bucket_11": "Art & design",
    "bucket_12": "Abstract & texture",
}


def load_manifests(export_dir: Path) -> dict[str, list[dict]]:
    """Build bucket groups from metadata.jsonl (always current; export manifests can be stale)."""
    meta = export_dir.parent / "metadata.jsonl"
    buckets: dict[str, list[dict]] = {}
    if not meta.exists():
        return buckets
    for line in meta.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("verdict") != "FINAL":
            continue
        b = rec.get("bucket", "unknown")
        buckets.setdefault(b, []).append(rec)
    return buckets


def resolve_servable_clip(export_dir: Path, rec: dict, bucket: str) -> Path | None:
    clip_id = rec["clip_id"]
    bucket_clip = export_dir / "by_bucket" / bucket / "clips" / f"{clip_id}.mp4"
    if bucket_clip.exists():
        return bucket_clip

    workspace_clip = export_dir.parent / "clips" / f"{clip_id}.mp4"
    if workspace_clip.exists():
        return link_clip_under_export(export_dir, workspace_clip)

    export_clip = export_dir / "clips" / f"{clip_id}.mp4"
    if export_clip.exists():
        return export_clip
    return None


def format_caption(rec: dict) -> str:
    return caption_for_review(rec) or "(no caption)"


def write_bucket_html(
    buckets: dict[str, list[dict]],
    export_dir: Path,
    out: Path,
    *,
    grid_cols: int = 2,
    grid_rows: int = 2,
) -> None:
    per_page = grid_cols * grid_rows
    bucket_ids = sorted(buckets.keys())
    total_clips = sum(len(v) for v in buckets.values())

    tab_buttons = []
    bucket_sections = []

    for bi, bucket in enumerate(bucket_ids):
        recs = buckets[bucket]
        label = BUCKET_LABELS.get(bucket, bucket)
        tab_buttons.append(
            f'<button class="tab{" active" if bi == 0 else ""}" data-bucket="{html.escape(bucket)}">'
            f'{html.escape(bucket)} · {html.escape(label)} ({len(recs)})</button>'
        )

        pages: list[str] = []
        page_cards: list[str] = []
        for rec in recs:
            clip_path = resolve_servable_clip(export_dir, rec, bucket)
            clip_rel = clip_url(out, clip_path) if clip_path else ""
            cap = html.escape(format_caption(rec))
            actors = html.escape(str(rec.get("clip_actors", [])))
            score = rec.get("final_score", "")
            page_cards.append(
                f"""
                <article class="card">
                  <div class="card-head">
                    <h2>{html.escape(rec['clip_id'])}</h2>
                    <p class="meta">{rec.get('timestamp_start', '?')}s–{rec.get('timestamp_end', '?')}s
                       · score {score} · actors {actors}</p>
                  </div>
                  <video controls muted loop playsinline preload="metadata"
                         src="{html.escape(clip_rel, quote=True)}"></video>
                  <pre class="caption">{cap}</pre>
                </article>
                """
            )
            if len(page_cards) == per_page:
                pages.append("".join(page_cards))
                page_cards = []
        if page_cards:
            while len(page_cards) < per_page:
                page_cards.append('<article class="card card-empty"></article>')
            pages.append("".join(page_cards))

        page_html = []
        for pi, grid in enumerate(pages):
            active = " active" if pi == 0 else ""
            page_html.append(
                f'<section class="bpage{active}" data-page="{pi}">'
                f'<div class="grid">{grid}</div></section>'
            )

        nav = ""
        if len(pages) > 1:
            nav = f"""
            <div class="bnav">
              <button class="bprev" type="button">← Prev</button>
              <span class="bindicator">1 / {len(pages)}</span>
              <button class="bnext" type="button">Next →</button>
            </div>"""

        active_bucket = " active" if bi == 0 else ""
        bucket_sections.append(
            f'<div class="bucket-panel{active_bucket}" data-bucket="{html.escape(bucket)}">'
            f'<h2 class="bucket-title">{html.escape(bucket)} — {html.escape(label)} ({len(recs)} clips)</h2>'
            f'{nav}'
            f'<div class="bucket-pages">{"".join(page_html)}</div>'
            f'</div>'
        )

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Bucket review ({total_clips} clips)</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  html, body {{
    height: 100%; margin: 0; overflow: hidden;
    font-family: system-ui, sans-serif; background: #111; color: #eee;
  }}
  header {{
    display: flex; flex-direction: column; gap: 0.35rem;
    padding: 0.5rem 1rem; border-bottom: 1px solid #333; background: #0d0d0d;
  }}
  header h1 {{ margin: 0; font-size: 1rem; font-weight: 600; }}
  .tabs {{ display: flex; flex-wrap: wrap; gap: 0.35rem; }}
  .tab {{
    background: #2a2a2a; color: #eee; border: 1px solid #444; border-radius: 4px;
    padding: 0.3rem 0.65rem; cursor: pointer; font-size: 0.78rem;
  }}
  .tab:hover {{ background: #383838; }}
  .tab.active {{ background: #3d5a80; border-color: #5a8fd4; }}
  main {{ height: calc(100vh - 4.5rem); overflow: hidden; position: relative; }}
  .bucket-panel {{ display: none; height: 100%; flex-direction: column; }}
  .bucket-panel.active {{ display: flex; }}
  .bucket-title {{ margin: 0.35rem 0.5rem; font-size: 0.85rem; color: #aaa; font-weight: 500; }}
  .bnav {{
    display: flex; gap: 0.5rem; align-items: center; justify-content: flex-end;
    padding: 0 0.5rem 0.25rem;
  }}
  .bnav button {{
    background: #2a2a2a; color: #eee; border: 1px solid #444; border-radius: 4px;
    padding: 0.25rem 0.7rem; cursor: pointer; font-size: 0.85rem;
  }}
  .bnav button:disabled {{ opacity: 0.35; }}
  .bindicator {{ color: #aaa; font-size: 0.85rem; min-width: 4rem; text-align: center; }}
  .bucket-pages {{ flex: 1; min-height: 0; position: relative; }}
  .bpage {{ display: none; height: 100%; width: 100%; }}
  .bpage.active {{ display: block; }}
  .grid {{
    display: grid; grid-template-columns: repeat({grid_cols}, 1fr);
    grid-template-rows: repeat({grid_rows}, 1fr);
    gap: 0.5rem; height: 100%; padding: 0.5rem;
  }}
  .card {{
    display: flex; flex-direction: column; min-height: 0; min-width: 0;
    border: 1px solid #333; border-radius: 6px; background: #1a1a1a; overflow: hidden;
  }}
  .card-empty {{ visibility: hidden; }}
  .card-head {{ flex: 0 0 auto; padding: 0.4rem 0.55rem 0; }}
  h2 {{ margin: 0; font-size: 0.82rem; font-weight: 600; }}
  .meta {{ color: #999; font-size: 0.72rem; margin: 0.15rem 0 0; }}
  video {{
    flex: 1 1 auto; min-height: 0; width: 100%; object-fit: contain;
    background: #000; border-top: 1px solid #2a2a2a; border-bottom: 1px solid #2a2a2a;
  }}
  .caption {{
    flex: 0 1 34%; min-height: 0; margin: 0; overflow-y: auto;
    white-space: pre-wrap; font-size: 0.72rem; line-height: 1.35;
    padding: 0.45rem 0.55rem; background: #141414;
  }}
</style></head><body>
<header>
  <h1>Bucket review · {total_clips} clips · {len(bucket_ids)} buckets · {grid_cols}×{grid_rows} grid</h1>
  <div class="tabs">{"".join(tab_buttons)}</div>
</header>
<main>
{"".join(bucket_sections)}
</main>
<script>
(function() {{
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".bucket-panel");

  function pauseAllVideos() {{
    document.querySelectorAll("video").forEach((v) => v.pause());
  }}

  tabs.forEach((tab) => {{
    tab.addEventListener("click", () => {{
      const bucket = tab.dataset.bucket;
      tabs.forEach((t) => t.classList.toggle("active", t === tab));
      panels.forEach((p) => p.classList.toggle("active", p.dataset.bucket === bucket));
      pauseAllVideos();
    }});
  }});

  panels.forEach((panel) => {{
    const pages = Array.from(panel.querySelectorAll(".bpage"));
    if (!pages.length) return;
    let idx = 0;
    const prev = panel.querySelector(".bprev");
    const next = panel.querySelector(".bnext");
    const indicator = panel.querySelector(".bindicator");

    function render() {{
      pages.forEach((p, i) => p.classList.toggle("active", i === idx));
      if (indicator) indicator.textContent = (idx + 1) + " / " + pages.length;
      if (prev) prev.disabled = idx === 0;
      if (next) next.disabled = idx === pages.length - 1;
      pauseAllVideos();
    }}

    if (prev) prev.addEventListener("click", () => {{ if (idx > 0) {{ idx--; render(); }} }});
    if (next) next.addEventListener("click", () => {{ if (idx < pages.length - 1) {{ idx++; render(); }} }});
    render();
  }});
}})();
</script>
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"Wrote {out} ({total_clips} clips, {len(bucket_ids)} buckets)")


def main() -> None:
    p = argparse.ArgumentParser(description="Bucket-grouped HTML review")
    p.add_argument("--workspace", required=True)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML path (default: <workspace>/export/bucket_review.html)",
    )
    p.add_argument("--grid-cols", type=int, default=2)
    p.add_argument("--grid-rows", type=int, default=2)
    args = p.parse_args()

    ws = Path(args.workspace).resolve()
    export_dir = ws / "export"
    if not export_dir.exists():
        raise SystemExit(f"Missing {export_dir}")

    buckets = load_manifests(export_dir)
    if not buckets:
        raise SystemExit("No bucket manifests found")

    out = args.output or (export_dir / "bucket_review.html")
    write_bucket_html(
        buckets, export_dir, out,
        grid_cols=args.grid_cols, grid_rows=args.grid_rows,
    )


if __name__ == "__main__":
    main()
