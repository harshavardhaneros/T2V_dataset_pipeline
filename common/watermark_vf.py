"""FFmpeg delogo filter for EROS / corner watermarks on exported clips."""

from __future__ import annotations

from typing import Any, Dict, Optional


def _delogo_rect(
    record: Dict[str, Any],
    export_cfg: Dict[str, Any],
    thresholds: Optional[Dict[str, Any]] = None,
) -> Optional[tuple[int, int, int, int]]:
    wm = record.get("watermark") or {}
    bbox = wm.get("bbox") or []
    if len(bbox) >= 4:
        x, y, w, h = (int(v) for v in bbox[:4])
        if w > 0 and h > 0:
            return x, y, w, h

    defaults = dict(export_cfg.get("watermark_delogo") or {})
    if thresholds:
        defaults.update(thresholds.get("watermark", {}).get("delogo") or {})
    if not defaults:
        return None
    x = int(defaults.get("x", 12))
    y = int(defaults.get("y", 10))
    w = int(defaults.get("w", 140))
    h = int(defaults.get("h", 58))
    return x, y, w, h


def should_remove_watermark(
    record: Dict[str, Any],
    export_cfg: Dict[str, Any],
) -> bool:
    if export_cfg.get("remove_watermark"):
        return True
    wm = record.get("watermark") or {}
    return bool(wm.get("present"))


def delogo_filter(
    record: Dict[str, Any],
    export_cfg: Dict[str, Any],
    thresholds: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if not should_remove_watermark(record, export_cfg):
        return None
    rect = _delogo_rect(record, export_cfg, thresholds)
    if not rect:
        return None
    x, y, w, h = rect
    return f"delogo=x={x}:y={y}:w={w}:h={h}"
