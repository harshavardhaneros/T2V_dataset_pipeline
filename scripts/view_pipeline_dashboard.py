#!/usr/bin/env python3
"""Build pipeline dashboard HTML: runtime, funnel, buckets, paginated clip browser."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.caption_text import caption_for_review  # noqa: E402
from common.review_clips import clip_url, link_frames_for_review, link_workspace_clips  # noqa: E402

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

SERVICE_LABELS = {
    "s1": "Scene detect & clip extract",
    "s2": "Motion / DOVER filter",
    "s3": "Letterbox removal",
    "s4": "Text detect (Gemma4 vLLM)",
    "s5": "Bucket classify",
    "s6": "Bucket verify",
    "s7": "Actor tagging",
    "s8": "Caption (vLLM)",
    "s9": "Quality scoring",
    "s10": "Verdict gate",
    "s11": "Export",
    "s12": "Report",
}

FUNNEL_STEPS = [
    ("s1", "clips_generated", "Extracted"),
    ("s2", "survivors", "After s2 filter"),
    ("s4", "text_present", "Text detected"),
    ("s5", "classified", "Classified"),
    ("s7", "tagged", "Actors tagged"),
    ("s8", "captioned", "Captioned"),
    ("s10", "FINAL", "FINAL verdict"),
    ("s11", "exported_clips", "Exported"),
]


def text_summary(
    records: list[dict],
    service_logs: dict[str, dict],
) -> dict[str, Any]:
    """Summarize s4 Gemma4 vLLM text detection."""
    s4_log = service_logs.get("s4", {})
    s4_stats = s4_log.get("stats", {})

    with_text = s4_stats.get("text_present")
    without_text = s4_stats.get("text_absent")
    if with_text is None:
        with_text = sum(1 for r in records if r.get("has_text"))
    if without_text is None:
        without_text = sum(1 for r in records if not r.get("has_text"))

    text_types = s4_stats.get("text_types")
    if not text_types:
        text_types = dict(
            Counter(
                str((r.get("text_overlay") or {}).get("text_type") or "other")
                for r in records
                if r.get("has_text")
            )
        )

    clips_processed = int(s4_stats.get("clips_updated", len(records)) or 0)
    runtime_seconds = float(s4_log.get("runtime_seconds", 0) or 0)
    model = s4_stats.get("model", "Gemma-4-31B-IT")
    backend = s4_stats.get("backend", "vllm")

    return {
        "runtime_seconds": runtime_seconds,
        "with_text": int(with_text),
        "without_text": int(without_text),
        "clips_processed": clips_processed,
        "text_types": text_types,
        "model": model,
        "backend": backend,
    }


def load_metadata(ws: Path) -> list[dict]:
    meta = ws / "metadata.jsonl"
    if not meta.exists():
        raise SystemExit(f"Missing {meta}")
    return [json.loads(line) for line in meta.read_text().splitlines() if line.strip()]


def load_service_runtimes(ws: Path, movie_stem: str) -> dict[str, dict]:
    logs = ws / "logs"
    out: dict[str, dict] = {}
    for i in range(1, 13):
        sid = f"s{i}"
        path = logs / f"s{i}" / f"{movie_stem}_runtime.json"
        if path.exists():
            out[sid] = json.loads(path.read_text(encoding="utf-8"))
    return out


def load_runtime_csv(ws: Path, video_id: str) -> dict[str, float]:
    path = ws / "reports" / "runtime_summary.csv"
    if not path.exists():
        return {}
    for line in path.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        if parts[0] == video_id and len(parts) >= 14:
            return {f"s{i}": float(parts[i] or 0) for i in range(1, 13)} | {
                "total": float(parts[13] or 0)
            }
    return {}


def fmt_seconds(sec: float) -> str:
    if sec >= 3600:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h}h {m}m {s:.0f}s"
    if sec >= 60:
        return f"{int(sec // 60)}m {sec % 60:.0f}s"
    return f"{sec:.1f}s"


def funnel_count(sid: str, stats: dict) -> int | None:
    if sid == "s10":
        return stats.get("FINAL")
    if sid == "s4":
        return stats.get("text_present") or stats.get("clips_updated")
    key = next((k for s, k, _ in FUNNEL_STEPS if s == sid), None)
    return stats.get(key) if key else None


def compact_clip(rec: dict) -> dict:
    cap = caption_for_review(rec) or ""
    actors = rec.get("clip_actors") or []
    bucket = rec.get("bucket") or "unknown"
    return {
        "id": rec["clip_id"],
        "bucket": bucket,
        "bucket_label": BUCKET_LABELS.get(bucket, bucket),
        "actors": actors,
        "actor_status": rec.get("actor_status") or "",
        "motion": rec.get("motion_score"),
        "uni": rec.get("unimatch_motion"),
        "vmaf": rec.get("vmaf_motion"),
        "dover": rec.get("dover_score"),
        "final": rec.get("final_score"),
        "verdict": rec.get("verdict") or "",
        "ts0": rec.get("timestamp_start"),
        "ts1": rec.get("timestamp_end"),
        "cap": cap,
        "tagged": bool(actors) or rec.get("actor_status") == "tagged",
        "has_text": bool(rec.get("has_text")),
        "text_type": str((rec.get("text_overlay") or {}).get("text_type") or ""),
    }


def write_serve_script(export_dir: Path, port: int, host: str) -> None:
  script = f"""#!/usr/bin/env bash
# Serve Devdas pipeline dashboard on the VM (bind all interfaces).
cd "$(dirname "$0")"
echo "Dashboard: http://{host}:{port}/pipeline_dashboard.html"
echo "Index:     http://{host}:{port}/index.html"
exec python3 -m http.server {port} --bind 0.0.0.0
"""
  path = export_dir / "serve.sh"
  path.write_text(script, encoding="utf-8")
  path.chmod(0o755)


def write_dashboard(
    ws: Path,
    export_dir: Path,
    records: list[dict],
    *,
    grid_cols: int = 4,
    grid_rows: int = 2,
    host: str = "101.53.140.144",
    port: int = 8765,
) -> Path:
    video_id = records[0]["video_id"] if records else ws.name
    movie_stem = Path(records[0].get("source_video", f"{video_id}.mp4")).stem if records else video_id

    service_logs = load_service_runtimes(ws, movie_stem)
    runtime_csv = load_runtime_csv(ws, video_id)

    report_path = ws / "reports" / f"{video_id}_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    buckets = Counter(r.get("bucket", "?") for r in records if r.get("keep", True))
    bucket_sorted = sorted(buckets.items(), key=lambda x: (-x[1], x[0]))
    max_bucket = max((c for _, c in bucket_sorted), default=1)

    tagged = sum(1 for r in records if r.get("actor_status") == "tagged" or r.get("clip_actors"))
    final = sum(1 for r in records if r.get("verdict") == "FINAL")
    kept = sum(1 for r in records if r.get("keep"))
    tx = text_summary(records, service_logs)

    runtime_rows = []
    max_rt = max((runtime_csv.get(f"s{i}", 0) for i in range(1, 13)), default=1) or 1
    total_rt = runtime_csv.get("total", sum(runtime_csv.get(f"s{i}", 0) for i in range(1, 13)))

    for i in range(1, 13):
        sid = f"s{i}"
        sec = runtime_csv.get(sid, 0)
        pct = (sec / max_rt * 100) if max_rt else 0
        stats = service_logs.get(sid, {}).get("stats", {})
        runtime_rows.append(
            f"""<tr>
              <td class="step">{html.escape(sid)}</td>
              <td class="name">{html.escape(SERVICE_LABELS.get(sid, sid))}</td>
              <td class="bar-cell"><div class="bar" style="width:{pct:.1f}%"></div></td>
              <td class="time">{html.escape(fmt_seconds(sec))}</td>
              <td class="detail">{html.escape(_runtime_detail(sid, stats))}</td>
            </tr>"""
        )

    funnel_items = []
    for sid, stat_key, label in FUNNEL_STEPS:
        stats = service_logs.get(sid, {}).get("stats", {})
        count = funnel_count(sid, stats)
        if count is None:
            continue
        funnel_items.append((label, count))
    max_funnel = max((c for _, c in funnel_items), default=1) or 1

    funnel_html = "".join(
        f"""<div class="funnel-row">
          <span class="funnel-label">{html.escape(label)}</span>
          <div class="funnel-bar-wrap"><div class="funnel-bar" style="width:{c / max_funnel * 100:.1f}%"></div></div>
          <span class="funnel-count">{c:,}</span>
        </div>"""
        for label, c in funnel_items
    )

    bucket_bars = "".join(
        f"""<button type="button" class="bucket-chip" data-bucket="{html.escape(b)}">
          <span class="bucket-name">{html.escape(BUCKET_LABELS.get(b, b))}</span>
          <span class="bucket-bar-wrap"><span class="bucket-bar" style="width:{c / max_bucket * 100:.1f}%"></span></span>
          <span class="bucket-count">{c}</span>
        </button>"""
        for b, c in bucket_sorted
    )

    actor_dist = report.get("actor_distribution") or {}
    actor_html = "".join(
        f'<div class="actor-stat"><b>{v:,}</b><span>{html.escape(k.replace("_", " "))}</span></div>'
        for k, v in sorted(actor_dist.items(), key=lambda x: -x[1])
    )

    tx_type_rows = "".join(
        f'<div class="wm-corner-row"><span>{html.escape(t.replace("_", " "))}</span><b>{n:,}</b></div>'
        for t, n in sorted(tx["text_types"].items(), key=lambda x: -x[1])
    ) or '<div class="wm-corner-row"><span>none</span><b>0</b></div>'
    s4_runtime = fmt_seconds(tx["runtime_seconds"]) if tx["runtime_seconds"] else "—"
    tx_panel = f"""
    <section class="panel wm-panel">
      <h2>Text detect (s4) — Gemma4 + vLLM</h2>
      <p class="panel-total">Per-clip middle frame · {html.escape(str(tx["backend"]))} · runtime <strong>{html.escape(s4_runtime)}</strong> · click a stat to filter</p>
      <div class="wm-hero wm-yes">
        <span class="wm-verdict">{tx["with_text"]:,} clips with on-screen text · {tx["without_text"]:,} without</span>
        <span class="wm-meta">Model: {html.escape(str(tx["model"]))}</span>
      </div>
      <div class="wm-stats" id="wm-stats">
        <button type="button" class="wm-stat wm-stat-btn" data-s4-filter="text"><b>{tx["with_text"]:,}</b><span>text detected</span></button>
        <button type="button" class="wm-stat wm-stat-btn" data-s4-filter="no-text"><b>{tx["without_text"]:,}</b><span>no text</span></button>
        <button type="button" class="wm-stat wm-stat-btn active" data-s4-filter=""><b>{tx["clips_processed"]:,}</b><span>all clips</span></button>
      </div>
      <h3 class="wm-sub">Text types</h3>
      <div class="wm-corners">{tx_type_rows}</div>
    </section>"""

    clips_sorted = sorted(records, key=lambda r: (r.get("bucket") or "", r.get("clip_id") or ""))
    clip_data = [compact_clip(r) for r in clips_sorted]
    per_page = grid_cols * grid_rows

    out = export_dir / "pipeline_dashboard.html"
    clip_json = json.dumps(clip_data, ensure_ascii=False)
    clip_prefix = "clips/"

    content = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(video_id)} — Pipeline Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
:root {{
  --bg: #0c0e14;
  --surface: #141820;
  --surface2: #1a2030;
  --border: #2a3348;
  --text: #e8ecf4;
  --muted: #8b95a8;
  --accent: #5b8def;
  --accent2: #7c5cff;
  --green: #3ecf8e;
  --amber: #f0b429;
  --rose: #f07178;
  --radius: 12px;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; font-family: "DM Sans", system-ui, sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.5;
  background-image: radial-gradient(ellipse 80% 50% at 50% -20%, rgba(91,141,239,.15), transparent);
}}
a {{ color: var(--accent); }}
.wrap {{ max-width: 1440px; margin: 0 auto; padding: 1.25rem 1.5rem 3rem; }}

.hero {{
  display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between;
  gap: 1rem; margin-bottom: 1.5rem; padding-bottom: 1.25rem; border-bottom: 1px solid var(--border);
}}
.hero h1 {{ margin: 0; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; }}
.hero .sub {{ color: var(--muted); font-size: 0.9rem; margin-top: 0.25rem; }}
.hero .vm {{ font-family: "JetBrains Mono", monospace; font-size: 0.8rem;
  background: var(--surface2); border: 1px solid var(--border); border-radius: 8px;
  padding: 0.5rem 0.85rem; color: var(--green); }}

.kpi-grid {{
  display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
  gap: 0.75rem; margin-bottom: 1.5rem;
}}
.kpi {{
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 1rem; text-align: center;
}}
.kpi b {{ display: block; font-size: 1.65rem; font-weight: 700; color: var(--accent); }}
.kpi span {{ font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}

.panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem; }}
@media (max-width: 960px) {{ .panels {{ grid-template-columns: 1fr; }} }}

.panel {{
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 1.1rem 1.25rem; overflow: hidden;
}}
.panel h2 {{ margin: 0 0 0.85rem; font-size: 0.95rem; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.06em; }}
.panel-total {{ font-size: 0.85rem; color: var(--muted); margin-bottom: 0.75rem; }}
.panel-total strong {{ color: var(--text); }}

.runtime-table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
.runtime-table td {{ padding: 0.35rem 0.4rem; vertical-align: middle; border-bottom: 1px solid #1e2433; }}
.runtime-table .step {{ font-family: "JetBrains Mono", monospace; color: var(--accent); width: 2rem; }}
.runtime-table .name {{ color: var(--muted); width: 38%; }}
.runtime-table .bar-cell {{ width: 28%; }}
.runtime-table .bar {{ height: 8px; border-radius: 4px;
  background: linear-gradient(90deg, var(--accent), var(--accent2)); min-width: 2px; }}
.runtime-table .time {{ font-family: "JetBrains Mono", monospace; white-space: nowrap; text-align: right; }}
.runtime-table .detail {{ font-size: 0.72rem; color: var(--muted); }}

.funnel-row {{ display: grid; grid-template-columns: 9rem 1fr 3.5rem; gap: 0.5rem;
  align-items: center; margin-bottom: 0.45rem; font-size: 0.82rem; }}
.funnel-label {{ color: var(--muted); }}
.funnel-bar-wrap {{ background: #1e2433; border-radius: 4px; height: 10px; overflow: hidden; }}
.funnel-bar {{ height: 100%; background: linear-gradient(90deg, var(--green), #2aa876); border-radius: 4px; }}
.funnel-count {{ font-family: "JetBrains Mono", monospace; text-align: right; }}

.bucket-section {{ margin-bottom: 1.5rem; }}
.bucket-chips {{ display: flex; flex-direction: column; gap: 0.4rem; }}
.bucket-chip {{
  display: grid; grid-template-columns: 11rem 1fr 3rem; gap: 0.6rem; align-items: center;
  background: var(--surface2); border: 1px solid var(--border); border-radius: 8px;
  padding: 0.45rem 0.75rem; cursor: pointer; text-align: left; color: var(--text);
  font: inherit; transition: border-color .15s, background .15s;
}}
.bucket-chip:hover, .bucket-chip.active {{ border-color: var(--accent); background: #1e2a40; }}
.bucket-name {{ font-size: 0.82rem; }}
.bucket-bar-wrap {{ background: #1e2433; border-radius: 4px; height: 8px; overflow: hidden; }}
.bucket-bar {{ display: block; height: 100%; background: linear-gradient(90deg, var(--accent2), var(--accent)); border-radius: 4px; }}
.bucket-count {{ font-family: "JetBrains Mono", monospace; font-size: 0.8rem; text-align: right; color: var(--muted); }}

.actor-grid {{ display: flex; flex-wrap: wrap; gap: 0.65rem; }}
.actor-stat {{ background: var(--surface2); border-radius: 8px; padding: 0.6rem 1rem; min-width: 100px; }}
.actor-stat b {{ display: block; font-size: 1.2rem; color: var(--amber); }}
.actor-stat span {{ font-size: 0.75rem; color: var(--muted); }}

.wm-panel .wm-hero {{
  padding: 0.85rem 1rem; border-radius: 8px; margin-bottom: 0.85rem;
  border: 1px solid var(--border); background: var(--surface2);
}}
.wm-panel .wm-hero.wm-yes {{ border-color: #3a5a40; background: rgba(62,207,142,.08); }}
.wm-panel .wm-hero.wm-no {{ border-color: #5a3a3a; background: rgba(240,113,120,.08); }}
.wm-verdict {{ display: block; font-weight: 600; font-size: 1rem; margin-bottom: 0.25rem; }}
.wm-meta {{ font-size: 0.8rem; color: var(--muted); }}
.wm-stats {{
  display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 0.65rem;
  margin-bottom: 0.85rem;
}}
.wm-stat {{
  background: var(--surface2); border: 1px solid var(--border); border-radius: 8px;
  padding: 0.65rem 0.85rem; text-align: center;
}}
.wm-stat-btn {{
  cursor: pointer; color: inherit; font: inherit; transition: border-color .15s, background .15s;
}}
.wm-stat-btn:hover, .wm-stat-btn.active {{
  border-color: var(--accent); background: #1e2a40;
}}
.wm-stat b {{ display: block; font-size: 1.35rem; color: var(--rose); }}
.wm-stat span {{ font-size: 0.72rem; color: var(--muted); }}
.wm-sub {{ margin: 0 0 0.5rem; font-size: 0.8rem; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
.wm-corners {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
.wm-corner-row {{
  display: flex; align-items: center; gap: 0.65rem; background: var(--surface2);
  border: 1px solid var(--border); border-radius: 8px; padding: 0.4rem 0.75rem; font-size: 0.82rem;
}}
.wm-corner-row b {{ font-family: "JetBrains Mono", monospace; color: var(--accent); }}
.wm-tag {{
  display: inline-block; font-size: 0.65rem; padding: 0.12rem 0.4rem; border-radius: 4px;
  background: rgba(240,113,120,.15); color: var(--rose); border: 1px solid rgba(240,113,120,.35);
  margin-top: 0.25rem;
}}

.clips-header {{
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem;
  margin: 2rem 0 1rem; padding-top: 1.5rem; border-top: 1px solid var(--border);
}}
.clips-header h2 {{ margin: 0; flex: 1; font-size: 1.15rem; }}
.filter-row {{ display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; }}
.filter-row select, .filter-row button {{
  background: var(--surface2); border: 1px solid var(--border); color: var(--text);
  border-radius: 8px; padding: 0.4rem 0.75rem; font: inherit; font-size: 0.85rem; cursor: pointer;
}}
.filter-row button.active {{ border-color: var(--accent); background: #1e2a40; }}
.nav-btns {{ display: flex; gap: 0.4rem; align-items: center; }}
.nav-btns button {{
  background: var(--surface2); border: 1px solid var(--border); color: var(--text);
  border-radius: 8px; padding: 0.45rem 1rem; cursor: pointer; font: inherit;
}}
.nav-btns button:disabled {{ opacity: 0.35; cursor: not-allowed; }}
#page-info {{ font-family: "JetBrains Mono", monospace; font-size: 0.85rem; color: var(--muted); min-width: 6rem; text-align: center; }}

.clip-grid {{
  display: grid; grid-template-columns: repeat({grid_cols}, 1fr);
  gap: 0.85rem; min-height: 480px;
}}
@media (max-width: 1200px) {{ .clip-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
@media (max-width: 640px) {{ .clip-grid {{ grid-template-columns: 1fr; }} }}

.clip-card {{
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  overflow: hidden; display: flex; flex-direction: column;
}}
.clip-card .head {{ padding: 0.65rem 0.85rem; border-bottom: 1px solid var(--border); }}
.clip-card .cid {{ font-family: "JetBrains Mono", monospace; font-size: 0.78rem; color: var(--accent); margin: 0; }}
.clip-card .bucket-tag {{
  display: inline-block; font-size: 0.72rem; font-weight: 600; margin-top: 0.25rem;
  padding: 0.15rem 0.5rem; border-radius: 999px;
  background: rgba(124,92,255,.2); color: #b8a8ff; border: 1px solid rgba(124,92,255,.35);
}}
.clip-card .meta {{ font-size: 0.72rem; color: var(--muted); margin: 0.3rem 0 0; }}
.clip-card .scores {{
  display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.4rem;
}}
.score-pill {{
  font-family: "JetBrains Mono", monospace; font-size: 0.68rem;
  background: var(--surface2); border-radius: 4px; padding: 0.12rem 0.4rem;
}}
.score-pill.motion {{ color: var(--amber); }}
.score-pill.dover {{ color: var(--green); }}
.score-pill.final {{ color: var(--accent); }}
.clip-card .actors {{
  font-size: 0.72rem; color: #9cb8a8; margin-top: 0.35rem;
}}
.clip-card video {{
  width: 100%; aspect-ratio: 16/9; background: #000; object-fit: contain; flex-shrink: 0;
}}
.clip-card .cap {{
  margin: 0; padding: 0.55rem 0.85rem; font-size: 0.72rem; line-height: 1.45;
  color: var(--muted); border-top: 1px solid var(--border);
  flex: 1 1 auto; min-height: 5rem; max-height: 11rem;
  overflow-x: hidden; overflow-y: scroll;
  white-space: pre-wrap; word-break: break-word;
  scrollbar-width: thin;
  scrollbar-color: var(--border) var(--surface2);
}}
.clip-card .cap::-webkit-scrollbar {{ width: 8px; }}
.clip-card .cap::-webkit-scrollbar-track {{ background: var(--surface2); border-radius: 4px; }}
.clip-card .cap::-webkit-scrollbar-thumb {{ background: #3a4a62; border-radius: 4px; }}
.clip-card .cap::-webkit-scrollbar-thumb:hover {{ background: var(--accent); }}
.empty-slot {{
  background: var(--surface); border: 1px dashed var(--border); border-radius: var(--radius);
  min-height: 200px;
}}
</style>
</head><body>
<div class="wrap">
  <header class="hero">
    <div>
      <h1>{html.escape(video_id)}</h1>
      <p class="sub">Indic video pipeline · {len(records):,} clips · {html.escape(movie_stem)}.mp4</p>
    </div>
    <div class="vm">http://{html.escape(host)}:{port}/pipeline_dashboard.html</div>
  </header>

  <div class="kpi-grid">
    <div class="kpi"><b>{len(records):,}</b><span>Total clips</span></div>
    <div class="kpi"><b>{kept:,}</b><span>Kept (s2)</span></div>
    <div class="kpi"><b>{final:,}</b><span>FINAL</span></div>
    <div class="kpi"><b>{tagged:,}</b><span>Actor tagged</span></div>
    <div class="kpi"><b>{tx["with_text"]:,}</b><span>Text in clip</span></div>
    <div class="kpi"><b>{len(bucket_sorted)}</b><span>Buckets used</span></div>
    <div class="kpi"><b>{html.escape(fmt_seconds(total_rt))}</b><span>Pipeline runtime</span></div>
  </div>

  <div class="panels">
    <section class="panel">
      <h2>Module runtime</h2>
      <p class="panel-total">Total pipeline time: <strong>{html.escape(fmt_seconds(total_rt))}</strong></p>
      <table class="runtime-table"><tbody>{"".join(runtime_rows)}</tbody></table>
    </section>
    <section class="panel">
      <h2>Clip funnel</h2>
      <p class="panel-total">Clips passing each stage</p>
      {funnel_html}
      <h2 style="margin-top:1.25rem">Actor tagging</h2>
      <div class="actor-grid">{actor_html or "<span class='muted'>No actor stats</span>"}</div>
    </section>
  </div>

  {tx_panel}

  <section class="panel bucket-section">
    <h2>Buckets — click to filter clips</h2>
    <div class="bucket-chips" id="bucket-chips">
      <button type="button" class="bucket-chip active" data-bucket="">All buckets · {len(records):,}</button>
      {bucket_bars}
    </div>
  </section>

  <div class="clips-header">
    <h2>Clip browser <small style="color:var(--muted);font-weight:400">({per_page} per page)</small></h2>
    <div class="filter-row">
      <select id="sort-by">
        <option value="bucket">Sort: bucket</option>
        <option value="id">Sort: clip id</option>
        <option value="dover-desc">Sort: DOVER ↓</option>
        <option value="motion-desc">Sort: motion ↓</option>
        <option value="tagged">Sort: tagged first</option>
        <option value="text-first">Sort: text first</option>
        <option value="no-text-first">Sort: no text first</option>
      </select>
      <button type="button" data-filter="all" class="active">All</button>
      <button type="button" data-filter="tagged">Tagged</button>
      <button type="button" data-filter="final">FINAL</button>
      <button type="button" data-s4-filter="text">Text</button>
      <button type="button" data-s4-filter="no-text">No text</button>
    </div>
    <div class="nav-btns">
      <button type="button" id="prev">← Prev</button>
      <span id="page-info">1 / 1</span>
      <button type="button" id="next">Next →</button>
    </div>
  </div>
  <div class="clip-grid" id="clip-grid"></div>
</div>

<script>
const CLIPS = {clip_json};
const PER_PAGE = {per_page};
const CLIP_PREFIX = {json.dumps(clip_prefix)};

let bucketFilter = "";
let tagFilter = "all";
let s4Filter = "";
let sortBy = "bucket";
let page = 0;

function sortClips(list) {{
  const copy = [...list];
  if (sortBy === "id") copy.sort((a,b) => a.id.localeCompare(b.id));
  else if (sortBy === "dover-desc") copy.sort((a,b) => (b.dover??0) - (a.dover??0));
  else if (sortBy === "motion-desc") copy.sort((a,b) => (b.motion??0) - (a.motion??0));
  else if (sortBy === "tagged") copy.sort((a,b) => (b.tagged?1:0)-(a.tagged?1:0) || a.id.localeCompare(b.id));
  else if (sortBy === "text-first") copy.sort((a,b) => (b.has_text?1:0)-(a.has_text?1:0) || a.id.localeCompare(b.id));
  else if (sortBy === "no-text-first") copy.sort((a,b) => (a.has_text?1:0)-(b.has_text?1:0) || a.id.localeCompare(b.id));
  else copy.sort((a,b) => (a.bucket||"").localeCompare(b.bucket||"") || a.id.localeCompare(b.id));
  return copy;
}}

function applyS4Filter(list) {{
  if (s4Filter === "text") return list.filter(c => c.has_text);
  if (s4Filter === "no-text") return list.filter(c => !c.has_text);
  return list;
}}

function setS4Filter(value) {{
  s4Filter = value || "";
  document.querySelectorAll("[data-s4-filter]").forEach(b => {{
    b.classList.toggle("active", (b.dataset.s4Filter || "") === s4Filter);
  }});
  page = 0;
  render();
}}

function filtered() {{
  let list = CLIPS;
  if (bucketFilter) list = list.filter(c => c.bucket === bucketFilter);
  if (tagFilter === "tagged") list = list.filter(c => c.tagged);
  if (tagFilter === "final") list = list.filter(c => c.verdict === "FINAL");
  list = applyS4Filter(list);
  return sortClips(list);
}}

function esc(s) {{
  const d = document.createElement("div");
  d.textContent = s ?? "";
  return d.innerHTML;
}}

function fmt(v, d=3) {{
  return v == null ? "—" : Number(v).toFixed(d);
}}

function render() {{
  const list = filtered();
  const totalPages = Math.max(1, Math.ceil(list.length / PER_PAGE));
  if (page >= totalPages) page = totalPages - 1;
  if (page < 0) page = 0;
  const slice = list.slice(page * PER_PAGE, (page + 1) * PER_PAGE);
  const grid = document.getElementById("clip-grid");
  grid.innerHTML = "";
  slice.forEach(c => {{
    const actors = (c.actors && c.actors.length) ? c.actors.join(", ") : (c.actor_status || "none");
    const el = document.createElement("article");
    el.className = "clip-card";
    el.innerHTML = `
      <div class="head">
        <p class="cid">${{esc(c.id)}}</p>
        <span class="bucket-tag">${{esc(c.bucket_label)}}</span>
        <p class="meta">${{fmt(c.ts0,1)}}s – ${{fmt(c.ts1,1)}}s · ${{esc(c.verdict)}}</p>
        <div class="scores">
          <span class="score-pill motion">motion ${{fmt(c.motion)}}</span>
          <span class="score-pill motion">uni ${{fmt(c.uni)}}</span>
          <span class="score-pill motion">vmaf ${{fmt(c.vmaf)}}</span>
          <span class="score-pill dover">dover ${{fmt(c.dover)}}</span>
          <span class="score-pill final">final ${{fmt(c.final)}}</span>
        </div>
        <div class="actors">Actors: ${{esc(actors)}}</div>
      </div>
      <video controls muted loop playsinline preload="metadata" src="${{CLIP_PREFIX}}${{esc(c.id)}}.mp4"></video>
      <p class="cap">${{esc(c.cap)}}</p>`;
    grid.appendChild(el);
  }});
  while (grid.children.length < PER_PAGE) {{
    const empty = document.createElement("div");
    empty.className = "empty-slot";
    grid.appendChild(empty);
  }}
  document.getElementById("page-info").textContent = (page+1) + " / " + totalPages + " (" + list.length + " clips)";
  document.getElementById("prev").disabled = page === 0;
  document.getElementById("next").disabled = page >= totalPages - 1;
}}

document.getElementById("bucket-chips").addEventListener("click", e => {{
  const btn = e.target.closest(".bucket-chip");
  if (!btn) return;
  bucketFilter = btn.dataset.bucket || "";
  document.querySelectorAll(".bucket-chip").forEach(b => b.classList.toggle("active", b === btn));
  page = 0;
  render();
}});

document.getElementById("wm-stats").addEventListener("click", e => {{
  const btn = e.target.closest("[data-s4-filter]");
  if (!btn) return;
  setS4Filter(btn.dataset.s4Filter || "");
}});

document.querySelector(".filter-row").addEventListener("click", e => {{
  const s4btn = e.target.closest("button[data-s4-filter]");
  if (s4btn) {{
    setS4Filter(s4btn.dataset.s4Filter || "");
    return;
  }}
  const btn = e.target.closest("button[data-filter]");
  if (!btn) return;
  tagFilter = btn.dataset.filter;
  document.querySelectorAll("button[data-filter]").forEach(b => b.classList.toggle("active", b === btn));
  page = 0;
  render();
}});

document.getElementById("sort-by").addEventListener("change", e => {{
  sortBy = e.target.value;
  page = 0;
  render();
}});

document.getElementById("prev").addEventListener("click", () => {{ page--; render(); }});
document.getElementById("next").addEventListener("click", () => {{ page++; render(); }});
document.addEventListener("keydown", e => {{
  if (e.key === "ArrowLeft") {{ page--; render(); }}
  if (e.key === "ArrowRight") {{ page++; render(); }}
}});

render();
</script>
</body></html>"""

    out.write_text(content, encoding="utf-8")
    return out


def _runtime_detail(sid: str, stats: dict) -> str:
    if not stats:
        return ""
    if sid == "s1":
        return f"{stats.get('clips_generated', '?')} clips"
    if sid == "s2":
        return f"kept {stats.get('kept', '?')}, dup {stats.get('duplicates_removed', 0)}"
    if sid == "s4":
        return (
            f"text {stats.get('text_present', '?')} · "
            f"no_text {stats.get('text_absent', '?')}"
        )
    if sid == "s5":
        return f"classified {stats.get('classified', '?')}"
    if sid == "s6":
        return f"verified {stats.get('verified_clips', '?')}"
    if sid == "s7":
        return f"tagged {stats.get('tagged', '?')}, no_match {stats.get('no_match', 0)}"
    if sid == "s8":
        return f"captioned {stats.get('captioned', '?')}"
    if sid == "s9":
        return f"scored {stats.get('scored', '?')}"
    if sid == "s10":
        return f"FINAL {stats.get('FINAL', 0)}"
    if sid == "s11":
        return f"exported {stats.get('exported_clips', '?')}"
    if sid == "s12":
        return f"final {stats.get('final_clips', '?')}"
    return ", ".join(f"{k}={v}" for k, v in list(stats.items())[:2])


def main() -> None:
    p = argparse.ArgumentParser(description="Build pipeline dashboard HTML")
    p.add_argument("--workspace", required=True)
    p.add_argument("--grid-cols", type=int, default=4)
    p.add_argument("--grid-rows", type=int, default=2)
    p.add_argument("--host", default="101.53.140.144", help="VM IP shown in header")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    ws = Path(args.workspace).resolve()
    export_dir = ws / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    records = load_metadata(ws)
    clip_ids = [r["clip_id"] for r in records]
    link_workspace_clips(export_dir, ws, clip_ids)
    link_frames_for_review(export_dir, ws, clip_ids)

    out = write_dashboard(
        ws,
        export_dir,
        records,
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
        host=args.host,
        port=args.port,
    )
    write_serve_script(export_dir, args.port, args.host)

    print(f"Wrote {out}")
    print(f"Serve: cd {export_dir} && ./serve.sh")
    print(f"Open:  http://{args.host}:{args.port}/pipeline_dashboard.html")


if __name__ == "__main__":
    main()
