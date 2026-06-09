"""Utilities for reading clip time ranges from metadata records."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def clip_time_range(record: Dict[str, Any]) -> Tuple[float, float]:
    return float(record["timestamp_start"]), float(record["timestamp_end"])
