"""Sample frames at fractional timestamps from a video clip."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from common.ffmpeg_utils import parse_crop_box


def sample_keyframes(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fractions: Optional[List[float]] = None,
    crop_box: str = "",
) -> List[np.ndarray]:
    """Return BGR frames at 25%, 50%, 75% of clip by default."""
    if fractions is None:
        fractions = [0.25, 0.5, 0.75]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    crop = parse_crop_box(crop_box)
    frames = []
    duration = end_sec - start_sec
    for frac in fractions:
        t = start_sec + duration * frac
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if ok and crop:
            cw, ch, cx, cy = crop
            frame = frame[cy : cy + ch, cx : cx + cw]
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def read_middle_frame(
    video_path: str,
    timestamp_start: float,
    timestamp_end: float,
    crop_box: str = "",
    time_offset: float = 0.0,
) -> Optional[np.ndarray]:
    start = timestamp_start - time_offset
    end = timestamp_end - time_offset
    mid = (start + end) / 2.0
    frames = sample_keyframes(
        video_path, start, end, fractions=[0.5], crop_box=crop_box
    )
    if frames:
        return frames[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid * fps))
    ok, frame = cap.read()
    cap.release()
    if ok and crop_box:
        crop = parse_crop_box(crop_box)
        if crop:
            cw, ch, cx, cy = crop
            frame = frame[cy : cy + ch, cx : cx + cw]
    return frame if ok else None
