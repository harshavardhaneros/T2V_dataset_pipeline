#!/usr/bin/env python3
"""HTML report: actor tagging vs names used in captions."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.caption_text import caption_to_str  # noqa: E402


def load_metadata(ws: Path) -> list[dict]:
    meta = ws / "metadata.jsonl"
    if not meta.exists():
        raise SystemExit(f"Missing {meta}")
    return [json.loads(line) for line in meta.read_text().splitlines() if line.strip()]


def caption_mentions_actor(rec: dict) -> bool:
    cap = caption_to_str(rec.get("caption")).lower()
    if not cap:
        return False
    return any(str(n).lower() in cap for n in (rec.get("clip_actors") or []))


def frame_img(export_dir: Path, clip_id: str) -> str:
    for idx in (2, 1, 3):
        p = export_dir / "frames" / f"{clip_id}.{idx}.jpg"
        if p.exists():
            return f"frames/{clip_id}.{idx}.jpg"
    return ""


def write_report(ws: Path, out: Path) -> dict[str, int]:
    export_dir = out.parent
    records = load_metadata(ws)

    tagged = [r for r in records if r.get("actor_status") == "tagged"]
    with_name = [r for r in tagged if caption_mentions_actor(r)]
    without_name = [r for r in tagged if not caption_mentions_actor(r)]

    name_hits = Counter()
    for r in with_name:
        cap = caption_to_str(r.get("caption")).lower()
        for n in r.get("clip_actors") or []:
            if str(n).lower() in cap:
                name_hits[str(n)] += 1

    status_counts = Counter(r.get("actor_status", "?") for r in records)
    video_id = records[0]["video_id"] if records else ws.name

    def card(rec: dict, matched: bool) -> str:
        clip_id = rec["clip_id"]
        actors = ", ".join(str(a) for a in (rec.get("clip_actors") or [])) or "—"
        cap = caption_to_str(rec.get("caption"))
        cap_short = cap if len(cap) <= 320 else cap[:317] + "…"
        img = frame_img(export_dir, clip_id)
        img_html = (
            f'<img src="{html.escape(img)}" alt="" loading="lazy"/>'
            if img
            else '<div class="no-img">no frame</div>'
        )
        badge = (
            '<span class="badge ok">name in caption</span>'
            if matched
            else '<span class="badge warn">tagged, name absent</span>'
        )
        pos = rec.get("pos_f2") or rec.get("pos_f1") or "—"
        sim = rec.get("actor_tag_min_similarity", "—")
        return f"""
<article class="card" data-match="{"yes" if matched else "no"}">
  <div class="thumb">{img_html}</div>
  <div class="body">
    <h3>{html.escape(clip_id)} {badge}</h3>
    <p class="actors"><b>clip_actors:</b> {html.escape(actors)}</p>
    <p class="meta">status={html.escape(str(rec.get("actor_status", "")))} · min_sim={html.escape(str(sim))}</p>
    <p class="meta">faces: {html.escape(str(pos))}</p>
    <p class="caption">{html.escape(cap_short)}</p>
  </div>
</article>"""

    cards_matched = "".join(card(r, True) for r in with_name)
    cards_unmatched = "".join(card(r, False) for r in without_name)
    name_lines = "".join(
        f"<li>{html.escape(name)}: <b>{cnt}</b> captions</li>"
        for name, cnt in name_hits.most_common()
    ) or "<li>none</li>"
    status_lines = "".join(
        f"<li>{html.escape(str(k))}: <b>{v}</b></li>"
        for k, v in sorted(status_counts.items())
    )

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>{html.escape(video_id)} — actor vs caption</title>
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"/>
<meta http-equiv="Pragma" content="no-cache"/>
<style>
  body {{ font-family: system-ui, sans-serif; background: #111; color: #eee;
    margin: 0; padding: 1rem 1.25rem 2rem; line-height: 1.45; }}
  a {{ color: #8cf; }}
  h1 {{ font-size: 1.35rem; margin: 0 0 0.25rem; }}
  .sub {{ color: #999; margin-bottom: 1rem; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 0.75rem; margin: 1rem 0; }}
  .stat {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 0.75rem; }}
  .stat b {{ display: block; font-size: 1.6rem; color: #7ab; }}
  .filters {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0; }}
  .filters button {{
    background: #222; color: #ccc; border: 1px solid #444; border-radius: 6px;
    padding: 0.4rem 0.75rem; cursor: pointer;
  }}
  .filters button.active {{ background: #2a4a6a; border-color: #5af; color: #fff; }}
  .grid {{ display: grid; gap: 0.75rem; }}
  .card {{
    display: grid; grid-template-columns: 200px 1fr; gap: 0.75rem;
    background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 0.75rem;
  }}
  .thumb img {{ width: 100%; border-radius: 4px; display: block; }}
  .no-img {{ width: 100%; height: 112px; background: #222; border-radius: 4px;
    display: flex; align-items: center; justify-content: center; color: #666; }}
  h3 {{ margin: 0 0 0.35rem; font-size: 0.95rem; }}
  .badge {{ font-size: 0.7rem; padding: 0.1rem 0.4rem; border-radius: 4px; margin-left: 0.25rem; }}
  .badge.ok {{ background: #1a3a2a; color: #8d8; }}
  .badge.warn {{ background: #3a2a1a; color: #da8; }}
  .actors {{ margin: 0.25rem 0; color: #bdf; }}
  .meta {{ margin: 0.15rem 0; color: #888; font-size: 0.85rem; }}
  .caption {{ margin: 0.35rem 0 0; color: #ccc; font-size: 0.9rem; }}
  .side {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0; }}
  ul {{ color: #bbb; margin: 0.25rem 0; padding-left: 1.2rem; }}
  .hidden {{ display: none !important; }}
  @media (max-width: 700px) {{
    .card {{ grid-template-columns: 1fr; }}
    .side {{ grid-template-columns: 1fr; }}
  }}
</style></head><body>
<p><a href="index.html">← review hub</a></p>
<h1>{html.escape(video_id)} — actor tagging vs captions</h1>
<p class="sub">Workspace: <code>{html.escape(str(ws))}</code></p>

<div class="stats">
  <div class="stat"><b>{len(records)}</b>total clips</div>
  <div class="stat"><b>{len(tagged)}</b>actor tagged</div>
  <div class="stat"><b>{len(with_name)}</b>tagged + name in caption</div>
  <div class="stat"><b>{len(without_name)}</b>tagged, name not in caption</div>
</div>

<div class="side">
  <div>
    <h2>actor_status</h2>
    <ul>{status_lines}</ul>
  </div>
  <div>
    <h2>names in captions (tagged clips)</h2>
    <ul>{name_lines}</ul>
  </div>
</div>

<div class="filters" id="filters">
  <button type="button" data-filter="all" class="active">All tagged ({len(tagged)})</button>
  <button type="button" data-filter="yes">Name in caption ({len(with_name)})</button>
  <button type="button" data-filter="no">Name absent ({len(without_name)})</button>
</div>

<div class="grid" id="cards">
{cards_matched}{cards_unmatched}
</div>

<script>
(() => {{
  let filter = "all";
  const cards = Array.from(document.querySelectorAll(".card"));
  function apply() {{
    cards.forEach((c) => {{
      const m = c.dataset.match;
      const show = filter === "all" || filter === m;
      c.classList.toggle("hidden", !show);
    }});
  }}
  document.getElementById("filters").addEventListener("click", (e) => {{
    const btn = e.target.closest("button[data-filter]");
    if (!btn) return;
    filter = btn.dataset.filter;
    document.querySelectorAll("#filters button").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    apply();
  }});
  apply();
}})();
</script>
</body></html>"""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return {
        "tagged": len(tagged),
        "with_name_in_caption": len(with_name),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build actor-vs-caption HTML report")
    p.add_argument("--workspace", required=True)
    p.add_argument(
        "--output",
        help="Output HTML path (default: workspace/export/actor_caption_report.html)",
    )
    args = p.parse_args()

    ws = Path(args.workspace).resolve()
    out = Path(args.output) if args.output else ws / "export" / "actor_caption_report.html"
    stats = write_report(ws, out)
    print(f"Wrote {out}")
    print(f"tagged clips: {stats['tagged']}")
    print(f"tagged + name in caption: {stats['with_name_in_caption']}")


if __name__ == "__main__":
    main()
