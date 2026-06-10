"""Pick transformers attention backend — avoid broken flash_attn wheels."""

from __future__ import annotations

import os


def resolve_attn_implementation() -> str:
    """Return flash_attention_2 only when the CUDA extension actually loads."""
    if os.environ.get("USE_FLASH_ATTN", "").lower() not in ("1", "true", "yes"):
        return "sdpa"
    try:
        import flash_attn_2_cuda  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"
