"""Efficient frame reads with direct timestamp seek."""

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
    """Read frames at given timestamps; seeks directly per timestamp (fast on long files)."""
    if not times_sec:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    crop = parse_crop_box(crop_box)
    out: List[np.ndarray] = []

    for t in times_sec:
        # Direct seek — avoids scanning from t=0 on 30+ min sources.
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t)) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(float(t) * fps)))
            ok, frame = cap.read()
        if not ok:
            continue
        if crop:
            cw, ch, cx, cy = crop
            frame = frame[cy : cy + ch, cx : cx + cw]
        if resize:
            w, h = resize
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        out.append(frame)

    cap.release()
    return out
