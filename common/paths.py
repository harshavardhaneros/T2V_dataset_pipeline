"""Resolve pipeline code root vs external outputs/models directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def pipeline_code_root(config: Dict[str, Any]) -> Path:
    return Path(config["pipeline"].get("pipeline_root", Path(__file__).resolve().parent.parent))


def outputs_root(config: Dict[str, Any]) -> Path:
    p = config["pipeline"].get("outputs_root")
    if not p:
        return pipeline_code_root(config)
    return Path(p)


def models_root(config: Dict[str, Any]) -> Path:
    p = config["pipeline"].get("models_root")
    if not p:
        return pipeline_code_root(config) / "models"
    return Path(p)


def _scoped_root(config: Dict[str, Any]) -> Path:
    run = config.get("_run")
    if run:
        return Path(run["root"])
    return outputs_root(config)


def workspaces_dir(config: Dict[str, Any]) -> Path:
    run = config.get("_run")
    if run:
        return Path(run["workspace"])
    rel = config["pipeline"].get("workspaces_dir", "workspaces")
    root = outputs_root(config)
    return root / rel if not Path(rel).is_absolute() else Path(rel)


def logs_dir(config: Dict[str, Any]) -> Path:
    rel = config["pipeline"].get("logs_dir", "logs")
    root = _scoped_root(config)
    return root / rel if not Path(rel).is_absolute() else Path(rel)


def reports_dir(config: Dict[str, Any]) -> Path:
    rel = config["pipeline"].get("reports_dir", "reports")
    root = _scoped_root(config)
    return root / rel if not Path(rel).is_absolute() else Path(rel)


def service_log_dir(config: Dict[str, Any], service_id: str) -> Path:
    n = service_id.replace("s", "")
    return logs_dir(config) / f"s{n}"


def qwen_model_path(config: Dict[str, Any]) -> Path:
    mp = config["pipeline"].get("master_pipeline", {})
    if mp.get("model_path"):
        return Path(mp["model_path"])
    return models_root(config) / "Qwen2.5-VL-32B-Instruct"


def yolo_face_model_path(config: Dict[str, Any]) -> Path:
    mp = config["pipeline"].get("master_pipeline", {})
    rel = mp.get("yolo_face_model", "yolov12n-face.pt")
    p = Path(rel)
    if p.is_absolute():
        return p
    # Prefer shared models dir, then Master actors/
    candidate = models_root(config) / "yolov12n-face.pt"
    if candidate.exists():
        return candidate
    root = Path(mp.get("root", ""))
    return root / rel if root else candidate
