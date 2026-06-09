"""Efficient sequential frame reads (avoid per-frame seek)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from common.ffmpeg_utils import parse_crop_box


def read_frames_at_times(
    video_path: str,
    times_sec: List[float],
    crop_box: str = "",
    resize: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """Read frames at given timestamps in one forward pass through the video."""
    if not times_sec:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    crop = parse_crop_box(crop_box)
    indexed = sorted(enumerate(times_sec), key=lambda x: x[1])
    out: List[Optional[np.ndarray]] = [None] * len(times_sec)
    frame_idx = 0

    for orig_i, t in indexed:
        target = max(0, int(t * fps))
        while frame_idx < target:
            if not cap.grab():
                break
            frame_idx += 1
        ok, frame = cap.read()
        if not ok:
            continue
        frame_idx += 1
        if crop:
            cw, ch, cx, cy = crop
            frame = frame[cy : cy + ch, cx : cx + cw]
        if resize:
            w, h = resize
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        out[orig_i] = frame

    cap.release()
    return [f for f in out if f is not None]
