"""Output directory layout: per-movie folders under outputs_root."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def output_layout(config: Dict[str, Any]) -> str:
    """per_movie | per_run | legacy"""
    pipeline = config.get("pipeline", {})
    if pipeline.get("output_layout"):
        return str(pipeline["output_layout"])
    if pipeline.get("isolate_run_outputs", False):
        return "per_run"
    return "legacy"


def run_output_enabled(config: Dict[str, Any]) -> bool:
    return output_layout(config) in ("per_movie", "per_run")


def outputs_root(config: Dict[str, Any]) -> Path:
    from common.paths import outputs_root as base_outputs_root

    return base_outputs_root(config)


def runs_base_dir(config: Dict[str, Any]) -> Path:
    rel = config["pipeline"].get("runs_dir", "runs")
    base = outputs_root(config)
    return base / rel if not Path(rel).is_absolute() else Path(rel)


def make_run_id(video_id: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{video_id}_{ts}"


def movie_output_dir(config: Dict[str, Any], video_id: str) -> Path:
    return outputs_root(config) / video_id


def find_movie_output(config: Dict[str, Any], video_id: str) -> Optional[Path]:
    layout = output_layout(config)
    if layout == "per_movie":
        path = movie_output_dir(config, video_id)
        return path if path.exists() else None
    if layout == "per_run":
        return find_latest_run(config, video_id)
    from common.paths import workspaces_dir

    path = workspaces_dir(config) / video_id
    return path if path.exists() else None


def find_latest_run(config: Dict[str, Any], video_id: str) -> Optional[Path]:
    base = runs_base_dir(config)
    if not base.exists():
        return None
    candidates = [
        p
        for p in base.iterdir()
        if p.is_dir() and p.name.startswith(f"{video_id}_")
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name, reverse=True)[0]


def init_run_output(
    config: Dict[str, Any],
    *,
    video_id: str,
    source_movie: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    new_run: bool = True,
) -> Path:
    """Bind logs/reports/workspace to one output directory for this movie/run."""
    layout = output_layout(config)

    if run_dir is not None:
        root = Path(run_dir).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Output directory not found: {root}")
        run_id = root.name
        workspace = root if layout == "per_movie" else root / "workspace"
    elif layout == "per_movie":
        root = movie_output_dir(config, video_id)
        root.mkdir(parents=True, exist_ok=True)
        run_id = video_id
        workspace = root
    elif layout == "per_run":
        if new_run:
            root = runs_base_dir(config) / make_run_id(video_id)
            root.mkdir(parents=True, exist_ok=False)
        else:
            latest = find_latest_run(config, video_id)
            if latest is None:
                raise FileNotFoundError(
                    f"No existing run found for video_id={video_id} under {runs_base_dir(config)}"
                )
            root = latest
        run_id = root.name
        workspace = root / "workspace"
    else:
        raise ValueError(f"init_run_output called with unsupported layout: {layout}")

    logs = root / "logs"
    reports = root / "reports"
    for d in (workspace, logs, reports):
        d.mkdir(parents=True, exist_ok=True)
    if layout == "per_movie":
        for sub in ("clips", "frames", "actor_frames", "actor_tags", "export"):
            (workspace / sub).mkdir(parents=True, exist_ok=True)
    for i in range(1, 13):
        (logs / f"s{i}").mkdir(parents=True, exist_ok=True)

    config["_run"] = {
        "run_id": run_id,
        "root": str(root),
        "video_id": video_id,
        "workspace": str(workspace),
        "layout": layout,
    }

    manifest = root / "run.json"
    manifest_payload = {
        "run_id": run_id,
        "video_id": video_id,
        "layout": layout,
        "source_movie": str(source_movie) if source_movie else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "logs": str(logs),
        "reports": str(reports),
    }
    if not manifest.exists():
        manifest_payload["created_at"] = manifest_payload["updated_at"]
    else:
        try:
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_payload["created_at"] = existing.get(
                "created_at", manifest_payload["updated_at"]
            )
        except json.JSONDecodeError:
            manifest_payload["created_at"] = manifest_payload["updated_at"]
    manifest.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    return root


def effective_output_root(config: Dict[str, Any]) -> Path:
    run = config.get("_run")
    if run:
        return Path(run["root"])
    return outputs_root(config)


def run_workspace_dir(config: Dict[str, Any]) -> Path:
    run = config.get("_run")
    if run:
        return Path(run["workspace"])
    from common.paths import workspaces_dir

    return workspaces_dir(config)
