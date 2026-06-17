#!/usr/bin/env python3
"""Generate full HTML review hub for a v3 workspace (clips, buckets, captions, actors, scores)."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.caption_text import caption_for_review  # noqa: E402
from common.review_clips import (  # noqa: E402
    clip_url,
    link_frames_for_review,
    link_workspace_clips,
)

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


def load_metadata(ws: Path) -> list[dict]:
    meta = ws / "metadata.jsonl"
    if not meta.exists():
        raise SystemExit(f"Missing {meta}")
    return [json.loads(line) for line in meta.read_text().splitlines() if line.strip()]


def actor_summary(rec: dict) -> str:
    parts: list[str] = []
    clip_actors = rec.get("clip_actors") or []
    if clip_actors:
        parts.append("clip: " + ", ".join(str(a) for a in clip_actors))
    for i in (1, 2, 3):
        af = rec.get(f"actors_f{i}") or []
        pos = rec.get(f"pos_f{i}", "unknown")
        if af:
            parts.append(f"f{i}: {af} @ {pos}")
    status = rec.get("actor_status", "")
    if status:
        parts.append(f"status={status}")
    return " · ".join(parts) if parts else "no actors"


def score_line(rec: dict) -> str:
    bits = []
    for key, label in (
        ("unimatch_motion", "uni"),
        ("vmaf_motion", "vmaf"),
        ("motion_score", "motion"),
        ("dover_score", "dover"),
        ("final_score", "final"),
        ("clip_score", "clip"),
    ):
        v = rec.get(key)
        if v is not None and v != "":
            bits.append(f"{label}={v}")
    return " · ".join(bits)


def status_badges(rec: dict) -> str:
    badges = []
    if rec.get("keep"):
        badges.append('<span class="badge ok">keep</span>')
    else:
        badges.append('<span class="badge bad">rejected</span>')
    verdict = rec.get("verdict") or "—"
    badges.append(f'<span class="badge">{html.escape(verdict)}</span>')
    bucket = rec.get("bucket") or "—"
    blabel = BUCKET_LABELS.get(bucket, bucket)
    badges.append(f'<span class="badge">{html.escape(blabel)}</span>')
    reason = rec.get("s2_reject_reason")
    if reason:
        badges.append(f'<span class="badge warn">{html.escape(reason)}</span>')
    if rec.get("actor_status") == "tagged":
        badges.append('<span class="badge actor">tagged</span>')
    return " ".join(badges)


def frame_thumbs(export_dir: Path, html_out: Path, clip_id: str) -> str:
    frames_dir = export_dir / "frames"
    imgs = []
    for idx in (1, 2, 3):
        p = frames_dir / f"{clip_id}.{idx}.jpg"
        if p.exists():
            rel = clip_url(html_out, p)
            imgs.append(f'<img src="{html.escape(rel, quote=True)}" alt="f{idx}" title="frame {idx}"/>')
    if not imgs:
        return ""
    return '<div class="frames">' + "".join(imgs) + "</div>"


def write_all_clips_html(
    records: list[dict],
    export_dir: Path,
    out: Path,
    *,
    grid_cols: int = 2,
    grid_rows: int = 2,
) -> None:
    per_page = grid_cols * grid_rows
    pages: list[str] = []
    page_cards: list[str] = []

    for rec in records:
        clip_id = rec["clip_id"]
        clip_path = export_dir / "clips" / f"{clip_id}.mp4"
        clip_rel = clip_url(out, clip_path) if clip_path.exists() else ""
        cap = html.escape(caption_for_review(rec) or "(no caption)")
        actors = html.escape(actor_summary(rec))
        scores = html.escape(score_line(rec))
        badges = status_badges(rec)
        thumbs = frame_thumbs(export_dir, out, clip_id)

        keep = "true" if rec.get("keep") else "false"
        tagged = "true" if rec.get("actor_status") == "tagged" or rec.get("clip_actors") else "false"
        final = "true" if rec.get("verdict") == "FINAL" else "false"
        rejected = "true" if not rec.get("keep", True) else "false"
        has_cap = "true" if rec.get("caption") or rec.get("generated_caption") else "false"

        video = (
            f'<video controls muted loop playsinline preload="metadata" src="{html.escape(clip_rel, quote=True)}"></video>'
            if clip_rel
            else '<p class="missing">clip not found</p>'
        )

        page_cards.append(
            f"""
            <article class="card" data-keep="{keep}" data-tagged="{tagged}"
                     data-final="{final}" data-rejected="{rejected}" data-caption="{has_cap}">
              <div class="card-head">
                <h2>{html.escape(clip_id)}</h2>
                <p class="meta">{rec.get('timestamp_start', '?')}s–{rec.get('timestamp_end', '?')}s
                   · {html.escape(rec.get('route') or '')}</p>
                <div class="badges">{badges}</div>
                <p class="scores">{scores}</p>
                <p class="actors">{actors}</p>
              </div>
              {thumbs}
              {video}
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
            f'<div class="grid">{grid_html}</div></section>'
        )

    total_pages = len(pages) or 1
    content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>All clips — {html.escape(records[0]['video_id'] if records else 'review')}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  html, body {{ height: 100%; margin: 0; overflow: hidden;
    font-family: system-ui, sans-serif; background: #111; color: #eee; }}
  header {{
    display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem;
    min-height: 3rem; padding: 0.5rem 1rem; border-bottom: 1px solid #333; background: #0d0d0d;
  }}
  header h1 {{ margin: 0; font-size: 1rem; flex: 1; }}
  .filters {{ display: flex; gap: 0.35rem; flex-wrap: wrap; }}
  .filters button {{ background: #222; color: #ccc; border: 1px solid #444; border-radius: 4px;
    padding: 0.3rem 0.65rem; cursor: pointer; font-size: 0.8rem; }}
  .filters button.active {{ background: #3a5a8a; color: #fff; border-color: #5a8ac8; }}
  .nav {{ display: flex; gap: 0.5rem; align-items: center; }}
  button.nav-btn {{ background: #2a2a2a; color: #eee; border: 1px solid #444; border-radius: 4px;
    padding: 0.35rem 0.85rem; cursor: pointer; }}
  button.nav-btn:disabled {{ opacity: 0.35; }}
  a {{ color: #7ab; }}
  main {{ height: calc(100vh - 4rem); position: relative; }}
  .page {{ display: none; height: 100%; }}
  .page.active {{ display: block; }}
  .grid {{
    display: grid; grid-template-columns: repeat({grid_cols}, 1fr);
    grid-template-rows: repeat({grid_rows}, 1fr);
    gap: 0.5rem; height: 100%; padding: 0.5rem;
  }}
  .card {{
    display: flex; flex-direction: column; min-height: 0; border: 1px solid #333;
    border-radius: 6px; background: #1a1a1a; overflow: hidden;
  }}
  .card.hidden {{ display: none !important; }}
  .card-head {{ padding: 0.4rem 0.6rem; border-bottom: 1px solid #2a2a2a; flex-shrink: 0; }}
  .card-head h2 {{ margin: 0; font-size: 0.85rem; }}
  .meta, .scores, .actors {{ margin: 0.15rem 0; font-size: 0.72rem; color: #aaa; }}
  .actors {{ color: #9cb; }}
  .badges {{ margin: 0.2rem 0; }}
  .badge {{ display: inline-block; font-size: 0.65rem; padding: 0.1rem 0.35rem;
    border-radius: 3px; background: #333; margin-right: 0.2rem; }}
  .badge.ok {{ background: #1a4a2a; }}
  .badge.bad {{ background: #4a1a1a; }}
  .badge.warn {{ background: #4a3a10; }}
  .badge.actor {{ background: #2a2a5a; }}
  .frames {{ display: flex; gap: 2px; padding: 0 0.4rem; flex-shrink: 0; }}
  .frames img {{ height: 48px; object-fit: cover; border-radius: 2px; }}
  video {{ width: 100%; flex: 1; min-height: 0; background: #000; object-fit: contain; }}
  .caption {{
    margin: 0; padding: 0.4rem 0.6rem; font-size: 0.68rem; line-height: 1.35;
    max-height: 28%; overflow: auto; background: #141414; color: #ccc;
    border-top: 1px solid #2a2a2a; white-space: pre-wrap; word-break: break-word;
  }}
  .missing {{ color: #a55; padding: 1rem; text-align: center; }}
</style></head><body>
<header>
  <h1>All clips · {len(records)} total · <a href="index.html">hub</a></h1>
  <div class="filters" id="filters">
    <button class="active" data-filter="all">All</button>
    <button data-filter="keep">Kept</button>
    <button data-filter="final">FINAL</button>
    <button data-filter="tagged">Actors tagged</button>
    <button data-filter="caption">Has caption</button>
    <button data-filter="rejected">Rejected</button>
  </div>
  <div class="nav">
    <button class="nav-btn" id="prev">Prev</button>
    <span id="page-indicator">1 / {total_pages}</span>
    <button class="nav-btn" id="next">Next</button>
  </div>
</header>
<main>{"".join(page_blocks)}</main>
<script>
(() => {{
  let filter = "all";
  const pages = Array.from(document.querySelectorAll(".page"));
  let idx = 0;

  function cardVisible(card) {{
    if (filter === "all") return true;
    if (filter === "keep") return card.dataset.keep === "true";
    if (filter === "final") return card.dataset.final === "true";
    if (filter === "tagged") return card.dataset.tagged === "true";
    if (filter === "caption") return card.dataset.caption === "true";
    if (filter === "rejected") return card.dataset.rejected === "true";
    return true;
  }}

  function applyFilter() {{
    document.querySelectorAll(".card:not(.card-empty)").forEach((c) => {{
      c.classList.toggle("hidden", !cardVisible(c));
    }});
  }}

  document.getElementById("filters").addEventListener("click", (e) => {{
    const btn = e.target.closest("button[data-filter]");
    if (!btn) return;
    filter = btn.dataset.filter;
    document.querySelectorAll("#filters button").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    applyFilter();
  }});

  function pauseAll() {{
    document.querySelectorAll("video").forEach((v) => {{ try {{ v.pause(); }} catch (_) {{}} }});
  }}

  function render() {{
    pages.forEach((p, i) => p.classList.toggle("active", i === idx));
    document.getElementById("page-indicator").textContent = (idx + 1) + " / " + pages.length;
    document.getElementById("prev").disabled = idx === 0;
    document.getElementById("next").disabled = idx === pages.length - 1;
    pauseAll();
    applyFilter();
  }}

  document.getElementById("prev").addEventListener("click", () => {{ if (idx > 0) {{ idx--; render(); }} }});
  document.getElementById("next").addEventListener("click", () => {{ if (idx < pages.length - 1) {{ idx++; render(); }} }});
  render();
}})();
</script>
</body></html>"""
    out.write_text(content, encoding="utf-8")
    print(f"Wrote {out} ({len(records)} clips, {total_pages} pages)")


def write_index_html(ws: Path, export_dir: Path, records: list[dict]) -> None:
    video_id = records[0]["video_id"] if records else ws.name
    n = len(records)
    kept = sum(1 for r in records if r.get("keep"))
    final = sum(1 for r in records if r.get("verdict") == "FINAL")
    tagged = sum(1 for r in records if r.get("actor_status") == "tagged" or r.get("clip_actors"))
    captioned = sum(1 for r in records if r.get("caption") or r.get("generated_caption"))
    buckets = Counter(r.get("bucket", "?") for r in records if r.get("keep"))
    reject_reasons = Counter(r.get("s2_reject_reason") for r in records if r.get("s2_reject_reason"))

    report_path = ws / "reports" / f"{video_id}_report.json"
    report_block = ""
    if report_path.exists():
        report = json.loads(report_path.read_text())
        report_block = "<h2>Pipeline report</h2><ul>" + "".join(
            f"<li><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</li>"
            for k, v in report.items()
            if k not in ("bucket_distribution", "actor_distribution")
        ) + "</ul>"

    bucket_lines = "".join(
        f"<li>{html.escape(BUCKET_LABELS.get(b, b))}: {c}</li>" for b, c in sorted(buckets.items())
    )
    reject_lines = "".join(
        f"<li>{html.escape(str(r))}: {c}</li>" for r, c in sorted(reject_reasons.items())
    )

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>{html.escape(video_id)} — review hub</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #111; color: #eee;
    max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1rem; color: #aaa; margin-top: 1.5rem; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 0.75rem; }}
  .stat {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 0.75rem; }}
  .stat b {{ display: block; font-size: 1.5rem; color: #7ab; }}
  .links {{ display: grid; gap: 0.5rem; margin: 1rem 0; }}
  a.card {{
    display: block; padding: 1rem 1.25rem; background: #1e2a3a; border: 1px solid #3a5a7a;
    border-radius: 8px; color: #cde; text-decoration: none; font-weight: 500;
  }}
  a.card:hover {{ background: #2a3a4a; }}
  a.card small {{ display: block; color: #89a; font-weight: 400; margin-top: 0.25rem; }}
  ul {{ color: #bbb; }}
  code {{ background: #222; padding: 0.15rem 0.4rem; border-radius: 3px; }}
</style></head><body>
<h1>{html.escape(video_id)}</h1>
<p>Workspace: <code>{html.escape(str(ws))}</code></p>

<div class="stats">
  <div class="stat"><b>{n}</b>clips total</div>
  <div class="stat"><b>{kept}</b>kept after s2</div>
  <div class="stat"><b>{final}</b>FINAL verdict</div>
  <div class="stat"><b>{tagged}</b>actor tagged</div>
  <div class="stat"><b>{captioned}</b>captioned</div>
</div>

<h2>Review pages</h2>
<div class="links">
  <a class="card" href="all_clips_review.html">All clips (full metadata)
    <small>Video + captions + motion/DOVER scores + actors + 3 frames · filterable</small></a>
  <a class="card" href="clip_review.html">Final clips (caption grid)
    <small>2×2 paginated — FINAL clips with captions only</small></a>
  <a class="card" href="bucket_review.html">Bucket review
    <small>Grouped by bucket · 2×2 grid per bucket</small></a>
  <a class="card" href="actor_caption_report.html">Actor vs caption
    <small>Tagged clips · which captions use actor names · filterable</small></a>
</div>

<h2>Data files</h2>
<ul>
  <li><a href="captions.jsonl">captions.jsonl</a></li>
  <li><a href="{html.escape(video_id)}_captions.csv">{video_id}_captions.csv</a></li>
  <li><a href="metadata.csv">metadata.csv</a></li>
  <li><a href="bucket_index.json">bucket_index.json</a></li>
</ul>

<h2>Buckets (kept clips)</h2>
<ul>{bucket_lines or "<li>none</li>"}</ul>

<h2>s2 reject reasons</h2>
<ul>{reject_lines or "<li>none</li>"}</ul>

{report_block}

<h2>Serve locally</h2>
<pre style="background:#1a1a1a;padding:1rem;border-radius:6px;overflow:auto">cd {html.escape(str(export_dir))}
python3 -m http.server 8765
# open http://localhost:8765/index.html</pre>
</body></html>"""
    out = export_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build full HTML review hub for a workspace")
    p.add_argument("--workspace", required=True)
    p.add_argument("--grid-cols", type=int, default=2)
    p.add_argument("--grid-rows", type=int, default=2)
    p.add_argument("--skip-subviews", action="store_true", help="Only build index + all_clips")
    args = p.parse_args()

    ws = Path(args.workspace).resolve()
    export_dir = ws / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    records = load_metadata(ws)
    clip_ids = [r["clip_id"] for r in records]

    link_workspace_clips(export_dir, ws, clip_ids)
    link_frames_for_review(export_dir, ws, clip_ids)

    write_index_html(ws, export_dir, records)
    write_all_clips_html(
        records,
        export_dir,
        export_dir / "all_clips_review.html",
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
    )

    scripts = _ROOT / "scripts"
    actor_report = scripts / "view_actor_caption_report.py"
    if actor_report.exists():
        cmd = [sys.executable, str(actor_report), "--workspace", str(ws)]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    if not args.skip_subviews:
        for script, extra in (
            ("view_clip_review.py", ["--skip-export"]),
            ("view_bucket_review.py", []),
        ):
            cmd = [sys.executable, str(scripts / script), "--workspace", str(ws), *extra]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

    print(f"\nServe: cd {export_dir} && python3 -m http.server 8765")
    print("Open:  http://localhost:8765/index.html")


if __name__ == "__main__":
    main()
