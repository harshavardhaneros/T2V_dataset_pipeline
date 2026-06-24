"""Resolve pipeline code root vs external outputs/models directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def pipeline_code_root(config: Dict[str, Any]) -> Path:
    return Path(config["pipeline"].get("pipeline_root", Path(__file__).resolve().parent.parent))


def master_pipeline_root(config: Dict[str, Any]) -> Path:
    """Resolve master/ actor-tagger + legacy captioner root (under pipeline_root)."""
    mp = config.get("pipeline", {}).get("master_pipeline", {})
    raw = mp.get("root", "master")
    path = Path(raw)
    if path.is_absolute():
        return path
    return pipeline_code_root(config) / path


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


def qwen_classify_model_path(config: Dict[str, Any]) -> Path:
    """7B classifier default (fast s5); falls back to 32B if 7B missing."""
    s5 = config.get("pipeline", {}).get("s5", {})
    mp = config.get("pipeline", {}).get("master_pipeline", {})
    explicit = s5.get("classify_model_path") or mp.get("classify_model_path")
    if explicit:
        return Path(explicit)
    root = models_root(config)
    for name in ("Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct"):
        candidate = root / name
        if (candidate / "config.json").exists():
            return candidate
    return root / "Qwen2.5-VL-7B-Instruct"


def video_caption_model_path(config: Dict[str, Any]) -> Path:
    """Resolve weights for native MP4 video captioning (s8 video backend)."""
    from common.caption_models import resolve_caption_model

    pcfg = config.get("pipeline", {}).get("captioner", {})
    if pcfg.get("caption_model") or pcfg.get("model_path"):
        resolved = resolve_caption_model(config)
        if resolved["backend"] == "video" or resolved["key"] in {
            "qwen2.5",
            "qwen3",
            "qwen3.5",
            "gemma4_dense",
        }:
            return resolved["model_path"]

    vc = config.get("models", {}).get("video_caption", {})
    qc = config.get("models", {}).get("qwen_video_caption", {})
    explicit = vc.get("model_path") or qc.get("model_path") or pcfg.get("model_path")
    if explicit:
        return Path(explicit)
    root = models_root(config)
    for name in (
        "Qwen3.5-27B",
        "Qwen2.5-VL-7B-Instruct",
        "Qwen2.5-VL-3B-Instruct",
        "Qwen3-VL-32B-Instruct",
        "gemma-4-31b-dense",
        "Qwen2.5-VL-32B-Instruct",
    ):
        candidate = root / name
        if (candidate / "config.json").exists():
            return candidate
    return root / "Qwen2.5-VL-7B-Instruct"


def qwen_video_model_path(config: Dict[str, Any]) -> Path:
    """Back-compat alias."""
    return video_caption_model_path(config)


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
