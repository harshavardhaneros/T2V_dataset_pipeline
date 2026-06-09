"""FFmpeg / ffprobe helpers."""

from __future__ import annotations

import re
import random
import subprocess
from typing import Optional, Tuple


def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    return float(subprocess.check_output(cmd).decode().strip())


def detect_crop(
    video_path: str,
    sample_seconds: int = 30,
    random_seed: int = 42,
) -> Optional[str]:
    """Return crop box as 'W:H:X:Y' or None."""
    duration = get_video_duration(video_path)
    random.seed(random_seed)
    max_start = max(0.0, duration - sample_seconds)
    sample_start = random.uniform(0, max_start) if max_start > 0 else 0.0

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-ss", str(sample_start),
        "-i", video_path,
        "-t", str(sample_seconds),
        "-vf", "cropdetect=24:16:0",
        "-f", "null",
        "-",
    ]
    result = subprocess.run(
        cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True
    )
    matches = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", result.stderr)
    if not matches:
        return None
    w, h, x, y = matches[-1]
    return f"{w}:{h}:{x}:{y}"


def parse_crop_box(crop_box: str) -> Optional[Tuple[int, int, int, int]]:
    if not crop_box:
        return None
    parts = crop_box.replace("crop=", "").split(":")
    if len(parts) != 4:
        return None
    return tuple(int(p) for p in parts)
