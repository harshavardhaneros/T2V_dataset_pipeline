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


def frame_position_label(actors: List[Dict[str, Any]], img_hw: tuple[int, int] | None) -> str:
    if not actors:
        return "unknown"
    h, w = img_hw if img_hw else (1080, 1920)
    parts = []
    for a in actors:
        if a.get("actor") in (None, "unknown"):
            continue
        name = a.get("display_name") or str(a.get("actor", "")).replace("_", " ").title()
        bbox = a.get("bbox")
        if bbox and len(bbox) >= 4:
            pos = screen_position(bbox, w, h)
            parts.append(f"{name} ({pos})")
        else:
            parts.append(name)
    return ", ".join(parts) if parts else "unknown"


def known_actor_names(actors: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for a in actors:
        if a.get("actor") in (None, "unknown"):
            continue
        n = a.get("display_name") or str(a.get("actor", "")).replace("_", " ").title()
        if n and n not in names:
            names.append(n)
    return names
