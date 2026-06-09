"""Map metadata timestamps to local video file time (trimmed segments)."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def clip_time_offset(record: Dict[str, Any], config: Dict[str, Any] | None = None) -> float:
    """Seconds subtracted from metadata times when reading the workspace video file."""
    if record.get("segment_time_offset_sec") is not None:
        return float(record["segment_time_offset_sec"])
    if config:
        return float(config.get("_test", {}).get("time_offset_sec", 0))
    return 0.0


def clip_local_range(
    record: Dict[str, Any],
    config: Dict[str, Any] | None = None,
) -> Tuple[float, float]:
    """Return (start, end) in seconds relative to the workspace video file."""
    off = clip_time_offset(record, config)
    return (
        float(record["timestamp_start"]) - off,
        float(record["timestamp_end"]) - off,
    )


def clip_local_middle(
    record: Dict[str, Any],
    config: Dict[str, Any] | None = None,
) -> float:
    start, end = clip_local_range(record, config)
    return (start + end) / 2.0
