"""Pickle-safe worker entry points for Ray / ProcessPool."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_DOVER_CLIENT = None


def _ensure_pipeline_root() -> None:
    root = os.environ.get("INDIC_PIPELINE_ROOT", "")
    if root and root not in sys.path:
        sys.path.insert(0, root)


def _clip_length_sec(config: Dict[str, Any]) -> float:
    return float(
        config.get("thresholds", {}).get("virtual_clips", {}).get("clip_length_sec", 5)
    )


def _motion_video_range(
    payload: Dict[str, Any],
) -> Tuple[str, float, float, str]:
    """Prefer short exported clip MP4 (0..clip_len) over full movie seek."""
    record = payload["record"]
    config = payload["config"]
    clip_path = payload.get("clip_path")
    if clip_path and Path(clip_path).exists() and Path(clip_path).stat().st_size > 0:
        return str(clip_path), 0.0, _clip_length_sec(config), ""

    from common.video_time import clip_local_range

    video_path = payload["video_path"]
    start, end = clip_local_range(record, config)
    return video_path, start, end, record.get("crop_box", "")


def score_clip_motion(payload: Dict[str, Any]) -> Dict[str, Any]:
    """UniMatch + VMAF motion for one clip."""
    _ensure_pipeline_root()
    from common.motion_filter import combine_motion_scores
    from common.motion_unimatch import compute_unimatch_motion
    from common.motion_vmaf import compute_vmaf_motion

    record = payload["record"]
    model_cfg = payload["model_cfg"]
    video_path, start, end, crop_box = _motion_video_range(payload)
    unimatch = compute_unimatch_motion(video_path, start, end, model_cfg, crop_box=crop_box)
    vmaf = compute_vmaf_motion(video_path, start, end, model_cfg, crop_box=crop_box)
    return {
        "clip_id": record["clip_id"],
        "unimatch_motion": round(unimatch, 4),
        "vmaf_motion": round(vmaf, 4),
        "motion_score": round(combine_motion_scores(unimatch, vmaf, model_cfg), 4),
    }


def extract_clip_frames_job(payload: Dict[str, Any]) -> int:
    """Extract 3 JPEG frames for one clip. Returns number of frames written."""
    _ensure_pipeline_root()
    from common.clip_io import extract_clip_frames

    video_path = Path(payload["video_path"])
    record = payload["record"]
    frames_dir = Path(payload["frames_dir"])
    return len(extract_clip_frames(video_path, record, frames_dir))


def vmaf_motion_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """CPU-only VMAF motion for one clip."""
    _ensure_pipeline_root()
    from common.motion_vmaf import compute_vmaf_motion

    record = payload["record"]
    model_cfg = payload["model_cfg"]
    video_path, start, end, crop_box = _motion_video_range(payload)
    vmaf = compute_vmaf_motion(video_path, start, end, model_cfg, crop_box=crop_box)
    return {"clip_id": record["clip_id"], "vmaf_motion": round(vmaf, 4)}


def export_clip_mp4_job(payload: Dict[str, Any]) -> str:
    """Export one clip MP4 via ffmpeg (CPU). Returns clip_id."""
    _ensure_pipeline_root()
    from common.clip_io import export_clip_mp4

    record = payload["record"]
    video_path = Path(payload["video_path"])
    clip_path = Path(payload["clip_path"])
    export_cfg = payload.get("export_cfg") or {}
    thresholds = payload.get("thresholds") or {}
    if clip_path.exists():
        try:
            if clip_path.stat().st_size > 0:
                return record["clip_id"]
        except OSError:
            pass
        clip_path.unlink(missing_ok=True)
    export_clip_mp4(
        video_path,
        record,
        clip_path,
        export_cfg=export_cfg,
        thresholds=thresholds,
    )
    return record["clip_id"]


def motion_scores_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compute unimatch + vmaf motion for one clip (CPU/GPU per worker config)."""
    return score_clip_motion(payload)


def dover_score_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """DOVER aesthetic/technical scores for one exported clip MP4."""
    _ensure_pipeline_root()
    global _DOVER_CLIENT
    from model_clients.dover_client import DoverClient

    record = payload["record"]
    clip_path = str(payload["clip_path"])
    if _DOVER_CLIENT is None:
        _DOVER_CLIENT = DoverClient(payload["config"])
    scores = _DOVER_CLIENT.score_video(clip_path)
    return {
        "clip_id": record["clip_id"],
        "aesthetic_score": round(float(scores["aesthetic_score"]), 4),
        "technical_score": round(float(scores["technical_score"]), 4),
        "dover_score": round(float(scores["dover_score"]), 4),
    }
