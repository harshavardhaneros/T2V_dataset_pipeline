"""Per-frame watermark corner and on-screen text heuristics (OpenCV, no OCR)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

CORNER_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")

TEMPLATE_SIZE = (80, 40)  # w, h — normalized logo patch size


def _wm_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("watermark", cfg.get("s4", {}).get("watermark", {}))


def _corner_bbox(
    frame_h: int,
    frame_w: int,
    corner: str,
    *,
    logo_w: int,
    logo_h: int,
) -> Tuple[int, int, int, int]:
    """Return x, y, w, h for a fixed-size corner crop."""
    lw = min(logo_w, frame_w)
    lh = min(logo_h, frame_h)
    if corner == "top_left":
        return 0, 0, lw, lh
    if corner == "top_right":
        return frame_w - lw, 0, lw, lh
    if corner == "bottom_left":
        return 0, frame_h - lh, lw, lh
    return frame_w - lw, frame_h - lh, lw, lh


def extract_corner_patch(
    frame: np.ndarray,
    corner: str,
    *,
    logo_w: int = 120,
    logo_h: int = 50,
) -> np.ndarray:
    h, w = frame.shape[:2]
    x, y, bw, bh = _corner_bbox(h, w, corner, logo_w=logo_w, logo_h=logo_h)
    return frame[y : y + bh, x : x + bw].copy()


def _corner_patches(
    frame: np.ndarray,
    *,
    logo_w: int = 120,
    logo_h: int = 50,
) -> Dict[str, np.ndarray]:
    return {
        corner: extract_corner_patch(frame, corner, logo_w=logo_w, logo_h=logo_h)
        for corner in CORNER_NAMES
    }


def _patch_features(patch: np.ndarray) -> Dict[str, float]:
    if patch is None or patch.size == 0:
        return {"std": 0.0, "edge_density": 0.0, "laplacian": 0.0, "mean": 0.0}

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    std = float(np.std(gray))
    mean = float(np.mean(gray))
    edges = cv2.Canny(gray, 60, 180)
    edge_density = float(np.mean(edges > 0))
    laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "std": std,
        "edge_density": edge_density,
        "laplacian": laplacian,
        "mean": mean,
    }


def _normalize_patch(patch: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    return cv2.resize(gray, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    if a.size != b.size or a.size == 0:
        return 0.0
    a -= a.mean()
    b -= b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(a, b) / denom)


def _heuristic_logo_score(feat: Dict[str, float], wm_cfg: Dict[str, Any]) -> float:
    """Score 0–1 for logo-like corner content; penalize bokeh (high std, low edges)."""
    std = feat["std"]
    edge = feat["edge_density"]
    lap = feat["laplacian"]

    min_std = float(wm_cfg.get("min_std", 18))
    max_std = float(wm_cfg.get("max_std", 90))
    min_edge = float(wm_cfg.get("min_edge_density", 0.10))
    min_lap = float(wm_cfg.get("min_laplacian", 80))

    if std < min_std or std > max_std:
        return 0.0
    if edge < min_edge or lap < min_lap:
        return 0.0
    # Bokeh / smooth highlights: bright blobs without edges
    if std > 55 and edge < 0.12:
        return 0.0

    std_score = min(1.0, (std - min_std) / max(1.0, max_std - min_std))
    edge_score = min(1.0, edge / max(min_edge, 0.01))
    lap_score = min(1.0, lap / max(min_lap, 1.0))
    return round(0.35 * std_score + 0.40 * edge_score + 0.25 * lap_score, 4)


def learn_logo_template(
    frames: List[np.ndarray],
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Find a corner with a stable overlay across sampled clips (broadcast logo).
    Returns template patch + corner name, or None if no consistent logo found.
    """
    wm_cfg = _wm_cfg(cfg)
    if not frames:
        return None

    logo_w = int(wm_cfg.get("logo_width", 120))
    logo_h = int(wm_cfg.get("logo_height", 50))
    min_stability = float(wm_cfg.get("template_min_stability", 0.62))
    min_template_edge = float(wm_cfg.get("template_min_edge_density", 0.08))

    best: Optional[Dict[str, Any]] = None
    for corner in CORNER_NAMES:
        patches = [
            _normalize_patch(extract_corner_patch(f, corner, logo_w=logo_w, logo_h=logo_h))
            for f in frames
            if f is not None and f.size > 0
        ]
        if len(patches) < 3:
            continue

        stack = np.stack(patches, axis=0).astype(np.float32)
        per_pixel_std = np.std(stack, axis=0)
        stability = 1.0 - min(1.0, float(np.mean(per_pixel_std)) / 40.0)
        template = np.median(stack, axis=0).astype(np.uint8)
        tmpl_feat = _patch_features(cv2.cvtColor(template, cv2.COLOR_GRAY2BGR))

        if stability < min_stability:
            continue
        if tmpl_feat["edge_density"] < min_template_edge:
            continue

        entry = {
            "corner": corner,
            "template": template,
            "stability": round(stability, 4),
            "template_edge_density": round(tmpl_feat["edge_density"], 4),
        }
        if best is None or entry["stability"] > best["stability"]:
            best = entry

    return best


def detect_corner_watermark(
    frame: np.ndarray,
    cfg: Dict[str, Any],
    *,
    template_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Detect a small fixed-corner broadcast logo using multi-feature heuristics
    and optional cross-clip template matching.
    """
    empty = {
        "present": False,
        "corner": None,
        "bbox": [],
        "mask_stored": False,
        "score": 0.0,
        "edge_density": 0.0,
        "template_match": 0.0,
        "method": "none",
    }
    if frame is None or frame.size == 0:
        return empty

    wm_cfg = _wm_cfg(cfg)
    logo_w = int(wm_cfg.get("logo_width", 120))
    logo_h = int(wm_cfg.get("logo_height", 50))
    template_threshold = float(wm_cfg.get("template_match_threshold", 0.52))
    heuristic_threshold = float(wm_cfg.get("heuristic_score_threshold", 0.45))

    h, w = frame.shape[:2]
    corner_results: Dict[str, Dict[str, Any]] = {}

    for corner in CORNER_NAMES:
        patch = extract_corner_patch(frame, corner, logo_w=logo_w, logo_h=logo_h)
        feat = _patch_features(patch)
        h_score = _heuristic_logo_score(feat, wm_cfg)
        tmpl_match = 0.0
        if template_info and template_info.get("template") is not None:
            norm = _normalize_patch(patch)
            tmpl_match = normalized_cross_correlation(norm, template_info["template"])

        corner_results[corner] = {
            "heuristic": h_score,
            "template_match": round(tmpl_match, 4),
            "feat": feat,
        }

    # Prefer template-guided detection when a stable movie logo was learned
    if template_info and template_info.get("template") is not None:
        corner = str(template_info["corner"])
        cr = corner_results.get(corner, {})
        tmpl_match = float(cr.get("template_match", 0))
        h_score = float(cr.get("heuristic", 0))
        present = tmpl_match >= template_threshold and h_score >= 0.25
        method = "template"
        score = tmpl_match
        if not present:
            # Strict fallback: any corner with very strong template + heuristic agreement
            for cname, cr2 in corner_results.items():
                tm = float(cr2.get("template_match", 0))
                hs = float(cr2.get("heuristic", 0))
                if tm >= template_threshold + 0.08 and hs >= 0.35:
                    present = True
                    corner = cname
                    score = tm
                    method = "template_fallback"
                    tmpl_match = tm
                    h_score = hs
                    break
    else:
        corner = max(corner_results, key=lambda c: corner_results[c]["heuristic"])
        cr = corner_results[corner]
        h_score = float(cr["heuristic"])
        tmpl_match = float(cr["template_match"])
        present = h_score >= heuristic_threshold
        score = h_score
        method = "heuristic"

    feat = corner_results.get(corner if present else "top_left", {}).get("feat", {})
    best_heuristic = max(float(corner_results[c]["heuristic"]) for c in CORNER_NAMES)
    x, y, bw, bh = _corner_bbox(h, w, corner if present else "top_left", logo_w=logo_w, logo_h=logo_h)
    final_score = score if present else best_heuristic

    return {
        "present": bool(present),
        "corner": corner if present else None,
        "bbox": [x, y, bw, bh] if present else [],
        "mask_stored": False,
        "score": round(float(final_score), 4),
        "edge_density": round(float(feat.get("edge_density", 0)), 4),
        "template_match": round(tmpl_match, 4),
        "heuristic_score": round(h_score if present else best_heuristic, 4),
        "method": method if present else "rejected",
    }


def detect_text_overlay(
    frame: np.ndarray,
    *,
    min_regions: int = 4,
    min_coverage: float = 0.004,
    min_region_area: int = 80,
    max_text_height_frac: float = 0.12,
    min_aspect: float = 0.8,
    max_aspect: float = 25.0,
) -> Dict[str, Any]:
    """Flag frames with subtitle / title / overlay text using morphology + contours."""
    if frame is None or frame.size == 0:
        return {"present": False, "regions": 0, "coverage": 0.0}

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    margin_h, margin_w = max(1, h // 5), max(1, w // 5)
    thresh[:margin_h, :] = 0
    thresh[h - margin_h :, :] = 0
    thresh[:, :margin_w] = 0
    thresh[:, w - margin_w :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_regions = 0
    total_area = 0
    for cnt in contours:
        _x, _y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < min_region_area:
            continue
        if bh > h * max_text_height_frac:
            continue
        aspect = bw / max(bh, 1)
        if aspect < min_aspect or aspect > max_aspect:
            continue
        text_regions += 1
        total_area += area

    coverage = total_area / float(h * w)
    present = text_regions >= min_regions or coverage >= min_coverage
    return {
        "present": bool(present),
        "regions": text_regions,
        "coverage": round(coverage, 5),
    }


def analyze_clip_frame(
    frame: Optional[np.ndarray],
    cfg: Dict[str, Any],
    *,
    template_info: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    text_cfg = cfg.get("text", cfg.get("s4", {}).get("text", {}))
    watermark = detect_corner_watermark(frame, cfg, template_info=template_info)
    text = detect_text_overlay(
        frame,
        min_regions=int(text_cfg.get("min_regions", 4)),
        min_coverage=float(text_cfg.get("min_coverage", 0.004)),
        min_region_area=int(text_cfg.get("min_region_area", 80)),
        max_text_height_frac=float(text_cfg.get("max_text_height_frac", 0.12)),
        min_aspect=float(text_cfg.get("min_aspect", 0.8)),
        max_aspect=float(text_cfg.get("max_aspect", 25.0)),
    )
    return watermark, text


def template_info_for_runtime(template_info: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Strip numpy template array for JSON logging."""
    if not template_info:
        return None
    return {
        "corner": template_info.get("corner"),
        "stability": template_info.get("stability"),
        "template_edge_density": template_info.get("template_edge_density"),
    }
