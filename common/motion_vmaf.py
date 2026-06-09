"""VMAF-style temporal difference motion scoring for clip filtering."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from common.ffmpeg_utils import parse_crop_box


def _sample_times(start_sec: float, end_sec: float, interval_sec: float) -> List[float]:
    times: List[float] = []
    t = start_sec
    while t < end_sec:
        times.append(t)
        t += interval_sec
    if len(times) < 2 and end_sec > start_sec:
        times.append(end_sec - 1e-3)
    return times


def _frame_diff_motion(
    video_path: str,
    start_sec: float,
    end_sec: float,
    crop_box: str = "",
    interval_sec: float = 0.5,
) -> float:
    from common.video_read import read_frames_at_times

    times = _sample_times(start_sec, end_sec, interval_sec)
    frames = read_frames_at_times(video_path, times, crop_box=crop_box)
    prev_gray: Optional[np.ndarray] = None
    diffs: List[float] = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            diffs.append(float(np.mean(diff)) / 255.0)
        prev_gray = gray
    return float(np.mean(diffs)) if diffs else 0.0


def _try_vmaf_cli(
    video_path: str,
    start_sec: float,
    end_sec: float,
    config: Dict[str, Any],
) -> Optional[float]:
    vmaf_bin = config.get("vmaf", {}).get("binary", "vmaf")
    if not shutil.which(vmaf_bin):
        return None
    duration = max(0.1, end_sec - start_sec)
    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "ref.y4m"
        dist = Path(tmp) / "dist.y4m"
        out = Path(tmp) / "vmaf.json"
        base_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start_sec:.3f}", "-i", video_path,
            "-t", f"{duration:.3f}",
        ]
        try:
            subprocess.run(base_cmd + ["-pix_fmt", "yuv420p", str(ref)], check=True, capture_output=True)
            subprocess.run(base_cmd + ["-pix_fmt", "yuv420p", str(dist)], check=True, capture_output=True)
            subprocess.run(
                [
                    vmaf_bin,
                    "-r", str(ref),
                    "-d", str(dist),
                    "-o", str(out),
                    "--json",
                ],
                check=True,
                capture_output=True,
            )
            payload = json.loads(out.read_text(encoding="utf-8"))
            pooled = payload.get("pooled_metrics", {})
            vmaf_score = pooled.get("vmaf", {}).get("mean")
            if vmaf_score is None:
                frames = payload.get("frames", [])
                if not frames:
                    return None
                vmaf_score = float(np.mean([f["metrics"]["vmaf"] for f in frames]))
            return float(max(0.0, min(1.0, 1.0 - float(vmaf_score) / 100.0)))
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError):
            return None


def compute_vmaf_motion(
    video_path: str,
    start_sec: float,
    end_sec: float,
    config: Dict[str, Any],
    crop_box: str = "",
) -> float:
    """Return normalized temporal-change score in [0, 1]."""
    cfg = config.get("vmaf", {})
    cli_score = _try_vmaf_cli(video_path, start_sec, end_sec, config)
    if cli_score is not None:
        return cli_score
    return _frame_diff_motion(
        video_path,
        start_sec,
        end_sec,
        crop_box=crop_box,
        interval_sec=float(cfg.get("sample_interval_sec", 0.5)),
    )
