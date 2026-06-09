"""Pickle-safe worker entry points for Ray / ProcessPool."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict


def _ensure_pipeline_root() -> None:
    root = os.environ.get("INDIC_PIPELINE_ROOT", "")
    if root and root not in sys.path:
        sys.path.insert(0, root)


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
    from common.video_time import clip_local_range

    record = payload["record"]
    config = payload["config"]
    video_path = payload["video_path"]
    model_cfg = payload["model_cfg"]
    start, end = clip_local_range(record, config)
    crop_box = record.get("crop_box", "")
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
    if clip_path.exists() and clip_path.stat().st_size > 0:
        return record["clip_id"]
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
    _ensure_pipeline_root()
    from common.motion_filter import combine_motion_scores
    from common.motion_unimatch import compute_unimatch_motion
    from common.motion_vmaf import compute_vmaf_motion
    from common.video_time import clip_local_range

    record = payload["record"]
    config = payload["config"]
    video_path = payload["video_path"]
    start, end = clip_local_range(record, config)
    crop_box = record.get("crop_box", "")
    model_cfg = payload["model_cfg"]
    unimatch = compute_unimatch_motion(video_path, start, end, model_cfg, crop_box=crop_box)
    vmaf = compute_vmaf_motion(video_path, start, end, model_cfg, crop_box=crop_box)
    return {
        "clip_id": record["clip_id"],
        "unimatch_motion": round(unimatch, 4),
        "vmaf_motion": round(vmaf, 4),
        "motion_score": round(combine_motion_scores(unimatch, vmaf, model_cfg), 4),
    }
