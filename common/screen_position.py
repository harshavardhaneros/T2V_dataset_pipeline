"""Screen position labels from face bounding boxes (eros-style)."""

from __future__ import annotations

from typing import Any, Dict, List


def screen_position(bbox: List[float], width: int, height: int) -> str:
    xc = (bbox[0] + bbox[2]) / 2
    yc = (bbox[1] + bbox[3]) / 2
    hp = "left" if xc < width * 0.33 else ("right" if xc >= width * 0.66 else "center")
    vp = "top" if yc < height * 0.33 else ("bottom" if yc >= height * 0.66 else "center")
    if hp == "center" and vp == "center":
        return "center"
    return f"{vp}-{hp}"


def frame_face_label(actors: List[Dict[str, Any]]) -> str:
    """Human-readable face list with pixel bbox (for metadata / review UI)."""
    if not actors:
        return "unknown"
    parts = []
    for a in actors:
        if a.get("actor") in (None, "unknown"):
            continue
        name = a.get("display_name") or str(a.get("actor", "")).replace("_", " ").title()
        bbox = a.get("bbox")
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = (int(v) for v in bbox[:4])
            parts.append(f"{name} bbox=[{x1},{y1},{x2},{y2}]")
        else:
            parts.append(name)
    return ", ".join(parts) if parts else "unknown"


def frame_position_label(actors: List[Dict[str, Any]], img_hw: tuple[int, int] | None) -> str:
    """Deprecated spatial labels; keep API but emit bbox-only labels."""
    return frame_face_label(actors)


def known_actor_names(actors: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for a in actors:
        if a.get("actor") in (None, "unknown"):
            continue
        n = a.get("display_name") or str(a.get("actor", "")).replace("_", " ").title()
        if n and n not in names:
            names.append(n)
    return names
