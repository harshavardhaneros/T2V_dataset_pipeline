"""UniMatch-style optical flow motion scoring for clip filtering."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from common.ffmpeg_utils import parse_crop_box


_UNIMATCH_CACHE: Dict[str, Any] = {}


def _sample_times(start_sec: float, end_sec: float, interval_sec: float) -> List[float]:
    times: List[float] = []
    t = start_sec
    while t < end_sec:
        times.append(t)
        t += interval_sec
    if len(times) < 2 and end_sec > start_sec:
        times.append(end_sec - 1e-3)
    return times


def _read_frames(
    video_path: str,
    times: List[float],
    crop_box: str = "",
    resize: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    from common.video_read import read_frames_at_times

    return read_frames_at_times(video_path, times, crop_box=crop_box, resize=resize)


def _mean_flow_magnitude(frames: List[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    magnitudes: List[float] = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        magnitudes.append(float(np.mean(mag)))
        prev_gray = gray
    return float(np.mean(magnitudes)) if magnitudes else 0.0


def _normalize_score(raw: float, scale: float = 8.0) -> float:
    return float(max(0.0, min(1.0, raw / scale)))


def _cache_key(cfg: Dict[str, Any]) -> str:
    device = str(cfg.get("device", "cuda"))
    ckpt = str(cfg.get("checkpoint", ""))
    repo = str(cfg.get("repo_path", ""))
    return f"{repo}|{ckpt}|{device}"


def _get_unimatch_model(cfg: Dict[str, Any]) -> Optional[Any]:
    """Load UniMatch once per process; reused across all clips."""
    repo_path = cfg.get("repo_path")
    ckpt = cfg.get("checkpoint")
    if not repo_path or not ckpt:
        return None
    root = Path(repo_path)
    if not root.exists() or not Path(ckpt).exists():
        return None

    key = _cache_key(cfg)
    if key in _UNIMATCH_CACHE:
        return _UNIMATCH_CACHE[key]

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import torch
        from unimatch.unimatch import UniMatch  # type: ignore

        device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        model = UniMatch(
            feature_channels=128,
            num_scales=2,
            upsample_factor=4,
            num_head=1,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=True,
            task="flow",
        ).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
        model.eval()
        _UNIMATCH_CACHE[key] = (model, device, torch)
        return _UNIMATCH_CACHE[key]
    except Exception:
        return None


def _try_unimatch_repo(
    frames: List[np.ndarray],
    config: Dict[str, Any],
) -> Optional[float]:
    cfg = config.get("unimatch", {})
    loaded = _get_unimatch_model(cfg)
    if loaded is None:
        return None
    model, device, torch = loaded
    try:
        mags: List[float] = []
        with torch.no_grad():
            for i in range(len(frames) - 1):
                img0 = torch.from_numpy(frames[i][:, :, ::-1]).permute(2, 0, 1).float()[None] / 255.0
                img1 = torch.from_numpy(frames[i + 1][:, :, ::-1]).permute(2, 0, 1).float()[None] / 255.0
                img0 = img0.to(device)
                img1 = img1.to(device)
                pred = model(
                    img0,
                    img1,
                    attn_type="swin",
                    attn_splits_list=[2],
                    corr_radius_list=[-1],
                    prop_radius_list=[-1],
                    num_reg_refine=1,
                )
                flow = pred["flow_preds"][-1][0].detach().cpu().numpy()
                mags.append(float(np.mean(np.sqrt(flow[0] ** 2 + flow[1] ** 2))))
        if not mags:
            return None
        return _normalize_score(float(np.mean(mags)), scale=float(cfg.get("scale", 8.0)))
    except Exception:
        return None


def compute_unimatch_motion(
    video_path: str,
    start_sec: float,
    end_sec: float,
    config: Dict[str, Any],
    crop_box: str = "",
) -> float:
    """Return normalized UniMatch-style motion score in [0, 1]."""
    cfg = config.get("unimatch", {})
    interval = float(cfg.get("sample_interval_sec", 0.5))
    resize = (
        int(cfg.get("resize_width", 576)),
        int(cfg.get("resize_height", 320)),
    )
    times = _sample_times(start_sec, end_sec, interval)
    frames = _read_frames(video_path, times, crop_box=crop_box, resize=resize)
    if len(frames) < 2:
        return 0.0

    repo_score = _try_unimatch_repo(frames, config)
    if repo_score is not None:
        return repo_score

    raw = _mean_flow_magnitude(frames)
    return _normalize_score(raw, scale=float(cfg.get("fallback_scale", 8.0)))
