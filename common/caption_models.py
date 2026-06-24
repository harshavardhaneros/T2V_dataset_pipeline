"""Caption model catalog — switch caption models and backends per run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, TypedDict

from common.paths import models_root

CAPTION_MODEL_ALIASES: Dict[str, str] = {
    "qwen2.5": "qwen2.5",
    "qwen2_5": "qwen2.5",
    "qwen-2.5": "qwen2.5",
    "qwen25": "qwen2.5",
    "qwen2.5-vl-7b": "qwen2.5",
    "qwen3": "qwen3",
    "qwen-3": "qwen3",
    "qwen3-vl-32b": "qwen3",
    "gemma3": "gemma3",
    "gemma-3": "gemma3",
    "gemma3-4b": "gemma3",
    "gemma-3-4b-it": "gemma3",
    "gemma4": "gemma4",
    "gemma-4": "gemma4",
    "gemma4-31b": "gemma4",
    "gemma-4-31b-it": "gemma4",
    "gemma4_dense": "gemma4_dense",
    "gemma4-dense": "gemma4_dense",
    "gemma-4-31b-dense": "gemma4_dense",
    "qwen3.5": "qwen3.5",
    "qwen3_5": "qwen3.5",
    "qwen-3.5": "qwen3.5",
    "qwen35": "qwen3.5",
    "qwen3.5-27b": "qwen3.5",
}


class CaptionModelSpec(TypedDict):
    key: str
    label: str
    family: str
    default_backend: str
    dir_name: str
    hf_repo: str
    model_path: Path
    backend: str


CAPTION_MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "qwen2.5": {
        "label": "Qwen2.5-VL-7B-Instruct",
        "family": "qwen",
        "default_backend": "vllm",
        "dir_name": "Qwen2.5-VL-7B-Instruct",
        "hf_repo": "Qwen/Qwen2.5-VL-7B-Instruct",
    },
    "qwen3": {
        "label": "Qwen3-VL-32B-Instruct",
        "family": "qwen",
        "default_backend": "vllm",
        "dir_name": "Qwen3-VL-32B-Instruct",
        "hf_repo": "Qwen/Qwen3-VL-32B-Instruct",
    },
    "gemma3": {
        "label": "Gemma-3-4B-IT",
        "family": "gemma",
        "default_backend": "gemma",
        "dir_name": "gemma-3-4b-it",
        "hf_repo": "google/gemma-3-4b-it",
    },
    "gemma4": {
        "label": "Gemma-4-31B-IT",
        "family": "gemma",
        "default_backend": "gemma",
        "dir_name": "gemma-4-31b-it",
        "hf_repo": "google/gemma-4-31B-it",
    },
    "gemma4_dense": {
        "label": "Gemma-4-31B-Dense-IT",
        "family": "gemma",
        "default_backend": "video",
        "dir_name": "gemma-4-31b-dense",
        "hf_repo": "google/gemma-4-31B-it",
    },
    "qwen3.5": {
        "label": "Qwen3.5-27B",
        "family": "qwen35",
        "default_backend": "video",
        "dir_name": "Qwen3.5-27B",
        "hf_repo": "Qwen/Qwen3.5-27B",
    },
}


def normalize_caption_backend(raw: str) -> str:
    """Normalize backend aliases (video clip input vs frame/vllm backends)."""
    backend = (raw or "").strip().lower()
    # Native video clips fed to vLLM (fast path for gemma4_dense).
    if backend in {"vllm_video", "video_vllm", "vllm-video", "native_vllm_video"}:
        return "vllm_video"
    # Native video clips fed to HuggingFace .generate() (slow path).
    if backend in {"qwen_video", "video_clip", "mp4", "native_mp4", "native_video"}:
        return "video"
    return backend


def normalize_caption_model_key(raw: str) -> str:
    key = (raw or "qwen2.5").strip().lower()
    key = CAPTION_MODEL_ALIASES.get(key, key)
    if key not in CAPTION_MODEL_SPECS:
        known = ", ".join(sorted(CAPTION_MODEL_SPECS))
        raise ValueError(f"Unknown caption_model {raw!r}. Choose one of: {known}")
    return key


def _infer_key_from_path(path: Path) -> str | None:
    name = path.name.lower()
    if "qwen3.5" in name or "qwen3_5" in name or "qwen35" in name:
        return "qwen3.5"
    if "qwen3" in name:
        return "qwen3"
    if "qwen2.5" in name or "qwen2_5" in name:
        return "qwen2.5"
    if "dense" in name and ("gemma-4" in name or "gemma4" in name):
        return "gemma4_dense"
    if "gemma-4" in name or "gemma4" in name:
        return "gemma4"
    if "gemma-3" in name or "gemma3" in name:
        return "gemma3"
    return None


def resolve_caption_model(config: Dict[str, Any]) -> CaptionModelSpec:
    """Resolve caption model key, local path, and backend for s8."""
    pcfg = config.get("pipeline", {}).get("captioner", {})
    explicit_path = pcfg.get("model_path")
    if pcfg.get("caption_model"):
        key = normalize_caption_model_key(pcfg["caption_model"])
    elif explicit_path:
        key = _infer_key_from_path(Path(explicit_path)) or "qwen2.5"
    else:
        key = "qwen2.5"

    spec = CAPTION_MODEL_SPECS[key]
    root = models_root(config)
    if explicit_path:
        model_path = Path(explicit_path)
    else:
        model_path = root / spec["dir_name"]

    backend = normalize_caption_backend(
        pcfg.get("backend") or spec["default_backend"]
    )
    return CaptionModelSpec(
        key=key,
        label=spec["label"],
        family=spec["family"],
        default_backend=spec["default_backend"],
        dir_name=spec["dir_name"],
        hf_repo=spec["hf_repo"],
        model_path=model_path,
        backend=backend,
    )


def caption_model_key(config: Dict[str, Any]) -> str:
    return resolve_caption_model(config)["key"]
