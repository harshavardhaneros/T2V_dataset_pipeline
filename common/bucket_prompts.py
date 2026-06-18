"""Load per-bucket caption prompts (video-adapted) for s8."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from common.master_bridge import bucket_to_category

BUCKET_FILE_PATTERN = re.compile(
    r"bucket_(\d{2})_([a-z0-9_]+)\.txt$", re.IGNORECASE
)


def resolve_bucket_prompts_dir(config: Dict[str, Any]) -> Path:
    """Resolve bucket prompt directory (pipeline_root/prompts preferred)."""
    pcfg = config.get("pipeline", {})
    mp = pcfg.get("master_pipeline", {})
    raw = mp.get("prompts_dir", "prompts")
    path = Path(raw)
    if path.is_absolute():
        return path
    proot = pcfg.get("pipeline_root")
    if proot:
        candidate = Path(proot) / path
        if candidate.is_dir():
            return candidate
    root = mp.get("root")
    if root:
        return Path(root) / path
    return Path("prompts")


@lru_cache(maxsize=4)
def _load_prompt_files(prompt_dir: str) -> Dict[str, str]:
    root = Path(prompt_dir)
    by_bucket: Dict[str, str] = {}
    by_slug: Dict[str, str] = {}
    if not root.is_dir():
        return {}
    for path in sorted(root.glob("bucket_*.txt")):
        m = BUCKET_FILE_PATTERN.match(path.name)
        if not m:
            continue
        text = path.read_text(encoding="utf-8").strip()
        bucket_id = f"bucket_{m.group(1)}"
        slug = m.group(2)
        by_bucket[bucket_id] = text
        by_slug[slug] = text
    out = dict(by_bucket)
    out.update(by_slug)
    out.update({bucket_to_category(k): v for k, v in by_bucket.items()})
    return out


def bucket_prompt_for_record(
    rec: Dict[str, Any],
    config: Dict[str, Any],
    *,
    prompt_mgr: Any = None,
) -> str:
    """Bucket-specific guidance text for a clip."""
    bucket = rec.get("bucket", "bucket_01")
    prompts = _load_prompt_files(str(resolve_bucket_prompts_dir(config)))
    if prompts:
        slug = ""
        if prompt_mgr:
            try:
                slug = prompt_mgr.get_bucket_info(bucket).get("slug", "")
            except KeyError:
                slug = ""
        category = bucket_to_category(bucket, slug)
        return (
            prompts.get(category)
            or prompts.get(bucket)
            or prompts.get("people_portraits", "")
        )
    if prompt_mgr:
        try:
            return prompt_mgr.get_prompt(bucket)
        except KeyError:
            return prompt_mgr.get_prompt("bucket_01")
    return ""
