#!/usr/bin/env python3
"""Export clip MP4s and build a 2x2 paginated HTML review (video + caption)."""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.caption_text import caption_for_review  # noqa: E402
from common.clip_io import export_clip_mp4  # noqa: E402
from common.review_clips import clip_url, link_clips_for_review  # noqa: E402
from common.video_files import find_movie_video  # noqa: E402


def load_records(meta_path: Path, only_final: bool, only_tagged: bool) -> list[dict]:
    rows = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
    if only_final:
        rows = [r for r in rows if r.get("verdict") == "FINAL" and r.get("keep")]
    if only_tagged:
        rows = [r for r in rows if r.get("actor_status") == "tagged"]
    return rows


def actor_label(rec: dict) -> str:
    actors = rec.get("actors") or []
    if not actors:
        return ""
    return ", ".join(a.get("display_name") or a.get("actor", "") for a in actors)


def find_source_video(ws: Path) -> Path | None:
    return find_movie_video(ws)


def write_clip_html(
    items: list[tuple[dict, str]],
    out: Path,
    *,
    grid_cols: int = 2,
    grid_rows: int = 2,
) -> None:
    per_page = grid_cols * grid_rows
    pages: list[list[str]] = []
    page_cards: list[str] = []

    for rec, clip_rel in items:
        cap = html.escape(caption_for_review(rec))
        actors = html.escape(actor_label(rec))
        page_cards.append(
            f"""
            <article class="card">
              <div class="card-head">
                <h2>{html.escape(rec['clip_id'])}</h2>
                <p class="meta">{rec.get('timestamp_start', '?')}s–{rec.get('timestamp_end', '?')}s
                   · {html.escape(rec.get('route') or '')} · {actors}</p>
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

    page_blocks = []
    for i, grid_html in enumerate(pages):
        active = " active" if i == 0 else ""
        page_blocks.append(
            f'<section class="page{active}" data-page="{i}">'
            f'<div class="grid" style="grid-template-columns:repeat({grid_cols},1fr);'
            f'grid-template-rows:repeat({grid_rows},1fr)">{grid_html}</div></section>'
        )

    total_pages = len(pages)
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Clip review ({len(items)})</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  html, body {{
    height: 100%; margin: 0; overflow: hidden;
    font-family: system-ui, sans-serif; background: #111; color: #eee;
  }}
  header {{
    display: flex; align-items: center; justify-content: space-between;
    height: 3rem; padding: 0 1rem; border-bottom: 1px solid #333; background: #0d0d0d;
  }}
  header h1 {{ margin: 0; font-size: 1rem; font-weight: 600; }}
  .nav {{ display: flex; gap: 0.5rem; align-items: center; }}
  button {{
    background: #2a2a2a; color: #eee; border: 1px solid #444; border-radius: 4px;
    padding: 0.35rem 0.85rem; cursor: pointer; font-size: 0.9rem;
  }}
  button:hover {{ background: #383838; }}
  button:disabled {{ opacity: 0.35; cursor: default; }}
  #page-indicator {{ min-width: 5rem; text-align: center; color: #aaa; font-size: 0.9rem; }}
  main {{ height: calc(100vh - 3rem); position: relative; }}
  .page {{ display: none; height: 100%; width: 100%; }}
  .page.active {{ display: block; }}
  .grid {{
    display: grid; gap: 0.5rem; height: 100%; width: 100%; padding: 0.5rem;
  }}
  .card {{
    display: flex; flex-direction: column; min-height: 0; min-width: 0;
    border: 1px solid #333; border-radius: 6px; background: #1a1a1a; overflow: hidden;
  }}
  .card-empty {{ visibility: hidden; }}
  .card-head {{ flex: 0 0 auto; padding: 0.4rem 0.55rem 0; }}
  h2 {{ margin: 0; font-size: 0.82rem; font-weight: 600; }}
  .meta {{ color: #999; font-size: 0.72rem; margin: 0.15rem 0 0; line-height: 1.3; }}
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
  <h1>Clip review · {len(items)} clips · {grid_cols}×{grid_rows} grid</h1>
  <div class="nav">
    <button id="prev" type="button">← Prev</button>
    <span id="page-indicator">1 / {total_pages}</span>
    <button id="next" type="button">Next →</button>
  </div>
</header>
<main>
{"".join(page_blocks)}
</main>
<script>
(function() {{
  const pages = Array.from(document.querySelectorAll(".page"));
  const prev = document.getElementById("prev");
  const next = document.getElementById("next");
  const indicator = document.getElementById("page-indicator");
  let idx = 0;

  function pauseOthers(activePage) {{
    document.querySelectorAll("video").forEach((v) => {{
      if (!activePage || !activePage.contains(v)) {{
        v.pause();
      }}
    }});
  }}

  function render() {{
    pages.forEach((p, i) => p.classList.toggle("active", i === idx));
    indicator.textContent = (idx + 1) + " / " + pages.length;
    prev.disabled = idx === 0;
    next.disabled = idx === pages.length - 1;
    pauseOthers(pages[idx]);
  }}

  prev.addEventListener("click", () => {{ if (idx > 0) {{ idx--; render(); }} }});
  next.addEventListener("click", () => {{ if (idx < pages.length - 1) {{ idx++; render(); }} }});
  window.addEventListener("keydown", (e) => {{
    if (e.key === "ArrowLeft" && idx > 0) {{ idx--; render(); }}
    if (e.key === "ArrowRight" && idx < pages.length - 1) {{ idx++; render(); }}
  }});
  render();
}})();
</script>
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"Wrote {out} ({len(items)} clips, {total_pages} pages)")


def main() -> None:
    p = argparse.ArgumentParser(description="Export clips + HTML video/caption review")
    p.add_argument("--workspace", required=True)
    p.add_argument("-n", "--num", type=int, default=0, help="Max clips (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--final-only", action="store_true", default=True)
    p.add_argument("--no-final-only", action="store_false", dest="final_only")
    p.add_argument("--tagged-only", action="store_true")
    p.add_argument("--clip-ids", nargs="*")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML path (default: <workspace>/export/clip_review.html)",
    )
    p.add_argument(
        "--clips-dir",
        type=Path,
        default=None,
        help="Clip MP4 dir (default: <workspace>/clips then export/clips)",
    )
    p.add_argument("--skip-export", action="store_true", help="Only rebuild HTML")
    p.add_argument("--force-export", action="store_true", help="Re-export all clips")
    p.add_argument("--grid-cols", type=int, default=2)
    p.add_argument("--grid-rows", type=int, default=2)
    args = p.parse_args()

    ws = Path(args.workspace).resolve()
    meta = ws / "metadata.jsonl"
    if not meta.exists():
        raise SystemExit(f"Missing {meta}")

    source = find_source_video(ws)
    if not source and not args.skip_export:
        raise SystemExit(f"No source video in {ws}")

    clips_dir = args.clips_dir or (ws / "clips")
    if not clips_dir.exists():
        clips_dir = ws / "export" / "clips"
    clips_dir = clips_dir.resolve()
    html_out = args.output or (ws / "export" / "clip_review.html")

    records = load_records(meta, args.final_only, args.tagged_only)
    if not records:
        raise SystemExit("No matching records for clip review")

    if args.clip_ids:
        by_id = {r["clip_id"]: r for r in records}
        picked = [by_id[c] for c in args.clip_ids if c in by_id]
    elif args.num > 0:
        picked = random.Random(args.seed).sample(records, min(args.num, len(records)))
    else:
        picked = records

    items: list[tuple[dict, str]] = []
    exported = skipped = failed = 0

    for rec in picked:
        clip_path = clips_dir / f"{rec['clip_id']}.mp4"
        if not args.skip_export:
            if clip_path.exists() and not args.force_export:
                skipped += 1
            else:
                ok = export_clip_mp4(
                    source,
                    rec,
                    clip_path,
                    export_cfg={"remove_watermark": True},
                )
                if ok:
                    exported += 1
                else:
                    failed += 1
                    continue
        elif not clip_path.exists():
            print(f"SKIP missing clip: {clip_path}")
            failed += 1
            continue

        items.append((rec, clip_path))

    if not items:
        raise SystemExit("No clips available for HTML review")

    export_dir = html_out.parent
    link_clips_for_review(export_dir, clips_dir, [rec["clip_id"] for rec, _ in items])
    html_items = [
        (rec, clip_url(html_out, export_dir / "clips" / f"{rec['clip_id']}.mp4"))
        for rec, _ in items
    ]
    write_clip_html(
        html_items,
        html_out,
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
    )
    print(f"Clips dir: {clips_dir}")
    print(f"exported={exported} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
