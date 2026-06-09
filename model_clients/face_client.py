"""Face / actor recognition placeholder."""

from __future__ import annotations

from typing import Any, Dict, List


class FaceClient:
    def __init__(self, config: Dict[str, Any]):
        self.use_placeholder = config.get("face", {}).get("use_placeholder", True)

    def detect_faces(self, image: Any) -> List[Dict[str, Any]]:
        return []
