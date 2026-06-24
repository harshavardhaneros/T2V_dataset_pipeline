#!/usr/bin/env python3
"""Copy Master bucket prompts and adapt them for short video-clip captioning."""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "master" / "prompts_legacy"
DST = ROOT / "prompts"

VIDEO_PROSE_RULES = """
────────────────────────────────────────────────────────────────────────────
VIDEO CLIP RULES (prose output)
────────────────────────────────────────────────────────────────────────────

- You are captioning a short VIDEO CLIP from sequential frames, not a still image.
- Write one rich paragraph in plain prose. No JSON, markdown, bullet lists, or headings.
- Use all frames to describe motion, actions, gestures, and camera movement over the clip.
- Start directly with the subject or action. Never open with phrases like
  "In the video clip", "In this clip", "The video shows", "This video depicts",
  "The clip shows", or "The image shows".
- Use exact Indian cultural names when visible (e.g. Kanjeevaram saree, dal makhani).
- Caption only what is visible; do not hallucinate.
- When cast names are provided in the user message, use those exact names for identified people.
"""

REPLACEMENTS = [
    (
        "You are a professional image annotator creating training data for a text-to-image model.",
        "You are a professional video-clip annotator creating training data for a video understanding dataset.",
    ),
    (
        "You will receive an image and a base description. Use both to generate a rich, accurate annotation.",
        "You will receive sequential frames from a short video clip. Use all frames to generate a rich, accurate description of the full clip.",
    ),
    ("This image belongs to", "This clip belongs to"),
    ("in the image", "in the clip"),
    ("in your caption:", "in your caption:"),
    ("Subjects or objects in the image", "Subjects, objects, and actions in the clip"),
    ("Image aesthetics", "Visual aesthetics"),
    ("Photographic style:", "Cinematography:"),
    ("this is critical for T2I training", "this is critical for video caption training"),
    ("Camera perspective:", "Camera movement and framing:"),
    (
        'Never start any sentence with "This image shows" or "The image depicts"',
        "Never use meta openers (see VIDEO CLIP RULES below)",
    ),
    ("Output ONLY valid JSON", "For structured JSON tasks only — prose captioning ignores JSON output"),
]


def adapt_text(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
  # Drop JSON schema / examples — prose pipeline uses system prompt for format.
    for marker in (
        "────────────────────────────────────────────────────────────────────────────\nOUTPUT SCHEMA",
        "────────────────────────────────────────────────────────────────────────────\nOUTPUT FORMAT",
        "OUTPUT FORMAT (return only this JSON",
    ):
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx].rstrip()
            break
    # Trim old GLOBAL RULES bullet-json instructions after marker
    text = re.sub(
        r"- The caption field must contain exactly 4 sentences.*?(?=\n\n|\Z)",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"────────────────+\nGLOBAL RULES\n─+\n\nGLOBAL RULES:\s*",
        "",
        text,
    )
    text = text.rstrip() + VIDEO_PROSE_RULES
    return text.strip() + "\n"


def main() -> int:
    if not SRC.is_dir():
        print(f"Source prompts missing: {SRC}", file=sys.stderr)
        return 1
    DST.mkdir(parents=True, exist_ok=True)
    for path in sorted(SRC.glob("bucket_*.txt")):
        out = DST / path.name
        adapted = adapt_text(path.read_text(encoding="utf-8"))
        out.write_text(adapted, encoding="utf-8")
        print(f"wrote {out.name}")
    print(f"Done: {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
