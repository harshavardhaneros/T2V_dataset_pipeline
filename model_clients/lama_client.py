"""LaMa inpainting placeholder."""

from __future__ import annotations

from typing import Any, Dict


class LamaClient:
    def __init__(self, config: Dict[str, Any]):
        self.use_placeholder = config.get("lama", {}).get("use_placeholder", True)

    def inpaint(self, image: Any, mask: Any) -> Any:
        return image
