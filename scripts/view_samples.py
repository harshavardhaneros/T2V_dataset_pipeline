#!/usr/bin/env python3
"""View pipeline samples: actor keyframe + caption side by side.

Writes an HTML gallery (default) or shows a matplotlib grid if a display is available.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import random
import sys
from pathlib import Path

# Allow imports when run from repo root or Indic_video_pipeline/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.caption_text import caption_to_str  # noqa: E402


def load_records(meta_path: Path, only_final: bool, only_tagged: bool) -> list[dict]:
    rows = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
    if only_final:
        rows = [r for r in rows if r.get("verdict") == "FINAL" and r.get("keep")]
    if only_tagged:
        rows = [r for r in rows if r.get("actor_status") == "tagged"]
    rows = [r for r in rows if r.get("caption")]
    return rows


def image_path(ws: Path, clip_id: str) -> Path:
    return ws / "actor_frames" / f"{clip_id}.jpg"


def actor_label(rec: dict) -> str:
    actors = rec.get("actors") or []
    if not actors:
        return ""
    return ", ".join(a.get("display_name") or a.get("actor", "") for a in actors)


def write_html(
    samples: list[tuple[dict, Path]],
    out: Path,
    *,
    grid_cols: int = 2,
    grid_rows: int = 2,
) -> None:
    per_page = grid_cols * grid_rows
    pages: list[list[str]] = []
    page_cards: list[str] = []

    for rec, img in samples:
        data = base64.b64encode(img.read_bytes()).decode("ascii")
        cap = html.escape(caption_to_str(rec.get("caption")))
        actors = html.escape(actor_label(rec))
        page_cards.append(
            f"""
            <article class="card">
              <div class="card-head">
                <h2>{html.escape(rec['clip_id'])}</h2>
                <p class="meta">{rec.get('timestamp_start', '?')}s–{rec.get('timestamp_end', '?')}s
                   · {html.escape(rec.get('route') or '')} · {actors}</p>
              </div>
              <img src="data:image/jpeg;base64,{data}" alt="{html.escape(rec['clip_id'])}"/>
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
<title>Pipeline samples ({len(samples)})</title>
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
  .card img {{
    flex: 1 1 auto; min-height: 0; width: 100%; object-fit: contain;
    background: #000; border-top: 1px solid #2a2a2a; border-bottom: 1px solid #2a2a2a;
  }}
  .caption {{
    flex: 0 1 38%; min-height: 0; margin: 0; overflow-y: auto;
    white-space: pre-wrap; font-size: 0.72rem; line-height: 1.35;
    padding: 0.45rem 0.55rem; background: #141414;
  }}
</style></head><body>
<header>
  <h1>Pipeline samples · {len(samples)} clips · {grid_cols}×{grid_rows} grid</h1>
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

  function render() {{
    pages.forEach((p, i) => p.classList.toggle("active", i === idx));
    indicator.textContent = (idx + 1) + " / " + pages.length;
    prev.disabled = idx === 0;
    next.disabled = idx === pages.length - 1;
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
    print(f"Wrote {out} ({len(samples)} samples, {total_pages} pages of {per_page})")


def show_matplotlib(samples: list[tuple[dict, Path]]) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(samples)
    cols = 1
    fig, axes = plt.subplots(n, cols, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]
    for ax, (rec, path) in zip(axes, samples):
        ax.imshow(Image.open(path))
        ax.axis("off")
        title = f"{rec['clip_id']} | {actor_label(rec)}"
        cap = caption_to_str(rec.get("caption"))
        if len(cap) > 500:
            cap = cap[:500] + "…"
        ax.set_title(title, fontsize=10, loc="left")
        ax.text(0, -0.02, cap, transform=ax.transAxes, fontsize=8, va="top", wrap=True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description="View actor frame + caption samples")
    p.add_argument(
        "--workspace",
        default="../pipeline_outputs/workspaces/devdas_standard",
        help="Path to workspace (metadata.jsonl + actor_frames/)",
    )
    p.add_argument("-n", "--num", type=int, default=6, help="Number of samples")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    p.add_argument("--final-only", action="store_true", default=True)
    p.add_argument("--no-final-only", action="store_false", dest="final_only")
    p.add_argument("--tagged-only", action="store_true", help="Only actor-tagged clips")
    p.add_argument(
        "--clip-ids",
        nargs="*",
        help="Explicit clip_ids (overrides random sample)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML output path (default: <workspace>/sample_review.html)",
    )
    p.add_argument("--matplotlib", action="store_true", help="Open matplotlib window")
    p.add_argument("--grid-cols", type=int, default=2, help="Grid columns per page (default 2)")
    p.add_argument("--grid-rows", type=int, default=2, help="Grid rows per page (default 2)")
    args = p.parse_args()

    ws = Path(args.workspace).resolve()
    meta = ws / "metadata.jsonl"
    if not meta.exists():
        raise SystemExit(f"Missing {meta}")

    records = load_records(meta, args.final_only, args.tagged_only)
    if not records:
        raise SystemExit("No records with captions match filters")

    if args.clip_ids:
        by_id = {r["clip_id"]: r for r in records}
        picked = [by_id[c] for c in args.clip_ids if c in by_id]
        missing = [c for c in args.clip_ids if c not in by_id]
        if missing:
            print("Unknown or filtered out:", ", ".join(missing))
    else:
        rng = random.Random(args.seed)
        picked = rng.sample(records, min(args.num, len(records)))

    samples: list[tuple[dict, Path]] = []
    for rec in picked:
        img = image_path(ws, rec["clip_id"])
        if not img.exists():
            print(f"SKIP (no image): {rec['clip_id']} -> {img}")
            continue
        samples.append((rec, img))

    if not samples:
        raise SystemExit("No actor_frames/*.jpg found for selected clips")

    out = args.output or (ws / "sample_review.html")
    write_html(samples, out, grid_cols=args.grid_cols, grid_rows=args.grid_rows)

    if args.matplotlib:
        show_matplotlib(samples)


if __name__ == "__main__":
    main()
