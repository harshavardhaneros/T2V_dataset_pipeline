"""Normalize caption field from VLM JSON to plain string."""

from __future__ import annotations

import json
import re
from typing import Any


def caption_to_str(caption: Any) -> str:
    """Coerce caption from str, list, or dict to a single string for CLIP/export."""
    if caption is None:
        return ""
    if isinstance(caption, str):
        text = caption.strip()
        if text.startswith("{") and "short_description" in text:
            try:
                obj = json.loads(text)
                if obj.get("short_description"):
                    return str(obj["short_description"]).strip()
            except json.JSONDecodeError:
                parsed = _loads_generated_caption(text)
                if parsed and parsed.get("short_description"):
                    return str(parsed["short_description"]).strip()
        return text
    if isinstance(caption, list):
        parts = [str(x).strip() for x in caption if x]
        return " ".join(parts)
    if isinstance(caption, dict):
        if caption.get("short_description"):
            return str(caption["short_description"]).strip()
        inner = caption.get("caption") or caption.get("text")
        return caption_to_str(inner)
    return str(caption).strip()


def _loads_generated_caption(text: str) -> dict[str, Any] | None:
    if not text or not str(text).strip():
        return None
    raw = str(text).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    if "{" in raw:
        start = raw.find("{")
        end = raw.rfind("}")
        if end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        m = re.search(r'"short_description"\s*:\s*"([^"]*)"', raw)
        if m:
            return {"short_description": m.group(1), "_truncated": True}
    return None


def prose_caption_for_export(rec: dict[str, Any]) -> str:
    """Plain prose caption for export files — never structured JSON."""
    struct = rec.get("caption_struct") or {}
    if isinstance(struct, dict) and struct.get("_format") == "prose":
        return caption_to_str(struct.get("short_description") or rec.get("caption"))

    for field in ("caption", "generated_caption"):
        text = caption_to_str(rec.get(field))
        if text and not (text.startswith("{") and "short_description" in text):
            return text

    for field in ("caption", "generated_caption"):
        parsed = _loads_generated_caption(rec.get(field) or "")
        if parsed and parsed.get("short_description"):
            return str(parsed["short_description"]).strip()

    return caption_to_str(rec.get("caption"))


def caption_for_review(rec: dict[str, Any]) -> str:
    """Full structured caption for HTML review (not just short_description)."""
    struct = rec.get("caption_struct") or {}
    if isinstance(struct, dict) and struct.get("_format") == "prose":
        return str(struct.get("short_description") or rec.get("caption") or "").strip()
    if isinstance(struct, dict) and struct and not struct.get("_parse_error"):
        return json.dumps(struct, indent=2, ensure_ascii=False)

    gen = rec.get("generated_caption", "")
    parsed = _loads_generated_caption(gen) if gen else None
    if parsed:
        return json.dumps(parsed, indent=2, ensure_ascii=False)

    text = caption_to_str(rec.get("caption"))
    if text.startswith("{"):
        loaded = _loads_generated_caption(text)
        if loaded:
            return json.dumps(loaded, indent=2, ensure_ascii=False)
    return text
