"""Load and map bucket prompts from external zip pack."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path
from typing import Dict, Optional


BUCKET_FILE_PATTERN = re.compile(
    r"bucket_(\d{2})_([a-z0-9_]+)\.txt$", re.IGNORECASE
)


class PromptManager:
    def __init__(self, zip_path: str, extract_dir: Optional[str] = None):
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            raise FileNotFoundError(f"Prompt pack not found: {zip_path}")
        if extract_dir:
            self.extract_dir = Path(extract_dir)
        else:
            digest = hashlib.md5(str(self.zip_path).encode()).hexdigest()[:12]
            self.extract_dir = self.zip_path.parent / f".prompt_cache_{digest}"
        self._registry: Dict[str, Dict[str, str]] = {}
        self._version: str = ""
        self._load()

    def _load(self) -> None:
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            zf.extractall(self.extract_dir)
            self._version = hashlib.sha256(self.zip_path.read_bytes()).hexdigest()[:16]

        for path in sorted(self.extract_dir.glob("bucket_*.txt")):
            m = BUCKET_FILE_PATTERN.match(path.name)
            if not m:
                continue
            num, slug = m.group(1), m.group(2)
            bucket_id = f"bucket_{num}"
            text = path.read_text(encoding="utf-8")
            if len(text.strip()) < 50:
                raise ValueError(f"Prompt too short: {path}")
            self._registry[bucket_id] = {
                "bucket_id": bucket_id,
                "slug": slug,
                "name": slug.replace("_", " "),
                "prompt": text,
                "file": path.name,
            }

        if len(self._registry) < 12:
            raise ValueError(
                f"Expected 12 bucket prompts, found {len(self._registry)}"
            )

    @property
    def version(self) -> str:
        return self._version

    @property
    def bucket_ids(self) -> list:
        return sorted(self._registry.keys())

    def get_prompt(self, bucket_id: str) -> str:
        if bucket_id not in self._registry:
            for bid, info in self._registry.items():
                if info["slug"] == bucket_id or info["name"] == bucket_id:
                    return info["prompt"]
            raise KeyError(f"Unknown bucket: {bucket_id}")
        return self._registry[bucket_id]["prompt"]

    def get_bucket_info(self, bucket_id: str) -> Dict[str, str]:
        if bucket_id not in self._registry:
            raise KeyError(f"Unknown bucket: {bucket_id}")
        return dict(self._registry[bucket_id])

    def slug_for_index(self, index: int) -> str:
        bid = f"bucket_{index:02d}"
        return self._registry[bid]["slug"]
