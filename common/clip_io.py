"""Extract per-clip MP4s and multi-frame JPEGs (eros-style offsets)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from common.ffmpeg_utils import parse_crop_box
from common.video_time import clip_local_range
from common.watermark_vf import delogo_filter

# Offsets within 3s clip: 0.5s, 1.5s, 2.5s (eros reference)
FRAME_OFFSETS = [0.5, 1.5, 2.5]


def clip_frame_path(frames_dir: Path, clip_id: str, idx: int) -> Path:
    return frames_dir / f"{clip_id}.{idx}.jpg"


def extract_clip_frames(
    video_path: Path,
    record: Dict[str, Any],
    frames_dir: Path,
    offsets: Optional[List[float]] = None,
) -> List[Path]:
    """Extract 3 cropped frames; also writes {clip_id}.jpg as middle frame."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    offsets = offsets or FRAME_OFFSETS
    start, _ = clip_local_range(record, None)
    crop = parse_crop_box(record.get("crop_box", ""))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    paths: List[Path] = []

    for i, offset in enumerate(offsets, 1):
        out = clip_frame_path(frames_dir, record["clip_id"], i)
        t = start + offset
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        if crop:
            cw, ch, cx, cy = crop
            frame = frame[cy : cy + ch, cx : cx + cw]
        cv2.imwrite(str(out), frame)
        paths.append(out)
        if i == 2:
            mid = frames_dir / f"{record['clip_id']}.jpg"
            cv2.imwrite(str(mid), frame)

    cap.release()
    return paths


def export_clip_mp4(
    source: Path,
    record: Dict[str, Any],
    out_path: Path,
    *,
    export_cfg: Optional[Dict[str, Any]] = None,
    thresholds: Optional[Dict[str, Any]] = None,
) -> bool:
    start, end = clip_local_range(record, None)
    duration = max(0.1, end - start)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vf_parts: list[str] = []
    crop = parse_crop_box(record.get("crop_box", ""))
    if crop:
        w, h, x, y = crop
        vf_parts.append(f"crop={w}:{h}:{x}:{y}")

    export_cfg = export_cfg or {}
    logo = delogo_filter(record, export_cfg, thresholds)
    if logo:
        vf_parts.append(logo)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-i", str(source),
        "-t", f"{duration:.3f}",
    ]
    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])
    cmd.extend([
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
        str(out_path),
    ])
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return out_path.exists() and out_path.stat().st_size > 0
    except subprocess.CalledProcessError:
        return False
