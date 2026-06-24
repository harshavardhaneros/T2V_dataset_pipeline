#!/usr/bin/env python3
"""
Shared constants and utilities for the master pipeline.

Centralises definitions that were previously duplicated across
pipeline.py, captioner.py, classifier.py,
actor_tagger.py, and frame_extractor.py.
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ── File extension sets ───────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv"}

# ── Model identifiers ────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"

# ── Shared helpers ────────────────────────────────────────────────────────────

def unique_stem(img_path: Path) -> str:
    """Collision-safe stem using ``parent__name`` convention.

    Every module that writes sidecar files (vlm_results, captions,
    actor_tags) must use the same naming so files can be cross-referenced.
    """
    return f"{img_path.parent.name}__{img_path.stem}"


def parse_llm_json(raw: str, fallback: dict | None = None) -> dict:
    """Parse LLM output that may be wrapped in markdown fences.

    Tries ``json.loads`` first, then extracts the first ``{…}`` block.
    Returns *fallback* (or ``{"_parse_error": True}``) on failure.
    """
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

    if fallback is not None:
        return fallback
    return {"_parse_error": True, "_raw": raw[:500]}
