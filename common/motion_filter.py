"""Motion score fusion and per-source percentile filtering."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def combine_motion_scores(
    unimatch_motion: float,
    vmaf_motion: float,
    config: Dict[str, Any],
) -> float:
    weights = config.get("motion", {})
    w_uni = float(weights.get("unimatch_weight", 0.7))
    w_vmaf = float(weights.get("vmaf_weight", 0.3))
    total = w_uni + w_vmaf
    if total <= 0:
        return 0.0
    return float((w_uni * unimatch_motion + w_vmaf * vmaf_motion) / total)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def motion_bounds_for_source(
    scores: List[float],
    config: Dict[str, Any],
) -> Tuple[float, float]:
    motion_cfg = config.get("motion", {})
    if motion_cfg.get("use_fixed_bounds", False):
        return (
            float(motion_cfg.get("min_score", 0.15)),
            float(motion_cfg.get("max_score", 0.80)),
        )
    if not scores:
        return (
            float(motion_cfg.get("min_score", 0.15)),
            float(motion_cfg.get("max_score", 0.80)),
        )
    p_low = float(motion_cfg.get("percentile_low", 10))
    p_high = float(motion_cfg.get("percentile_high", 90))
    lo = _percentile(scores, p_low / 100.0)
    hi = _percentile(scores, p_high / 100.0)
    floor = float(motion_cfg.get("min_score_floor", 0.05))
    ceiling = float(motion_cfg.get("max_score_ceiling", 0.95))
    return max(floor, lo), min(ceiling, hi)


def classify_motion_failure(
    motion_score: float,
    lower: float,
    upper: float,
) -> str:
    if motion_score < lower:
        return "static"
    if motion_score > upper:
        return "excessive_motion"
    return ""
