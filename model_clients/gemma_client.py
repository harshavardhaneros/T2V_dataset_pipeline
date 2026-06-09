"""Gemma verifier client (placeholder for future swap)."""

from __future__ import annotations

from typing import Any, Dict


class GemmaClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("gemma", {})
        self.use_placeholder = self.config.get("use_placeholder", True)

    def verify(self, prompt: str) -> Dict[str, Any]:
        return {"verified": True, "confidence": 0.9, "route": "other"}
