"""Qwen3-VL client (vLLM endpoint) with placeholder fallback."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

import requests


class QwenClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("qwen", config)
        self.endpoint = self.config.get("endpoint", "http://localhost:8000/v1")
        self.model_name = self.config.get("model_name", "qwen3-vl")
        self.timeout = self.config.get("timeout_sec", 120)
        self.use_placeholder = self.config.get("use_placeholder", True)

    def _placeholder_response(self, task: str, prompt: str, seed: str = "") -> str:
        h = int(hashlib.md5((task + seed + prompt[:200]).encode()).hexdigest(), 16)
        if task == "classify":
            buckets = [f"bucket_{i:02d}" for i in range(1, 13)]
            return json.dumps({
                "bucket": buckets[h % 12],
                "bucket_confidence": 0.75 + (h % 25) / 100,
                "reject": False,
                "reject_reason": None,
            })
        if task == "verify":
            route = "people" if h % 3 == 0 else "other"
            return json.dumps({
                "verified": True,
                "confidence": 0.85 + (h % 15) / 100,
                "route": route,
            })
        if task == "caption":
            actor_line = ""
            if "IMPORTANT: The person(s) in this image are:" in prompt:
                import re
                m = re.search(
                    r"IMPORTANT: The person\(s\) in this image are: ([^.]+)\.",
                    prompt,
                )
                if m:
                    names = m.group(1).strip()
                    actor_line = (
                        f"• {names} visible in the scene with natural posture and expression. "
                    )
            if not actor_line:
                actor_line = (
                    "• Subjects and actions visible in the cultural scene. "
                )
            return json.dumps({
                "caption": (
                    actor_line
                    + "• Indoor or outdoor Indian setting with contextual background. "
                    "• Warm natural lighting with balanced colour and calm mood. "
                    "• Mid-shot framing with subjects centred in focus."
                ),
                "tags": {"setting": "outdoor", "lighting": "natural daylight"},
            })
        return "{}"

    def complete(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        task: str = "generic",
        seed: str = "",
    ) -> str:
        if self.use_placeholder:
            return self._placeholder_response(task, prompt, seed)

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
        }
        try:
            r = requests.post(
                f"{self.endpoint}/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            return self._placeholder_response(task, prompt, seed)
