"""vLLM batched multi-frame captioning for s8."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from common.bucket_prompts import bucket_prompt_for_record
from common.gemma_caption import (
    build_caption_user_text,
    get_caption_system_prompt,
    pick_caption_frames,
)
from common.clip_io import frame_offsets_for_record
from common.caption_models import resolve_caption_model
from common.qwen_vllm import QwenVLLMEngine


def _images_for_clip(
    rec: Dict[str, Any],
    frames_dir: Path,
    actor_frames_dir: Path,
    clip_path: Path,
) -> List[Image.Image]:
    frame_paths = pick_caption_frames(rec, frames_dir)
    if not frame_paths:
        frame_paths = pick_caption_frames(rec, actor_frames_dir)
    if frame_paths:
        return [Image.open(p).convert("RGB") for p in frame_paths]

    import cv2

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = float(rec.get("duration", 5))
    offsets = [duration * f for f in (0.2, 0.5, 0.8)]
    images: List[Image.Image] = []
    for t in offsets:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok:
            images.append(Image.fromarray(frame[:, :, ::-1]))
    cap.release()
    return images


def _caption_prompt_parts(
    rec: Dict[str, Any], config: Dict[str, Any]
) -> tuple[str, str | None]:
    """Return (user_text, system_text). system_text is set for Gemma family."""
    resolved = resolve_caption_model(config)
    offsets = frame_offsets_for_record(rec, config)
    bucket_guidance = bucket_prompt_for_record(
        rec, config, prompt_mgr=config.get("_prompt_manager")
    )
    user = build_caption_user_text(
        rec,
        multi_frame=True,
        frame_offsets=offsets,
        bucket_guidance=bucket_guidance,
    )
    if resolved["family"] == "gemma":
        return user, get_caption_system_prompt(config)
    from common.qwen_video_caption import build_video_caption_prompt

    return build_video_caption_prompt(rec, config), None


def caption_clips_vllm(
    config: Dict[str, Any],
    items: List[Tuple[Dict[str, Any], Path]],
    *,
    frames_dir: Path,
    actor_frames_dir: Path,
    engine: Optional[QwenVLLMEngine] = None,
) -> List[str]:
    own_engine = engine is None
    if own_engine:
        engine = QwenVLLMEngine.acquire(config, stage="s8")
    assert engine is not None

    batch: List[tuple] = []
    order: List[Dict[str, Any]] = []

    for rec, clip_path in items:
        images = _images_for_clip(rec, frames_dir, actor_frames_dir, clip_path)
        if not images:
            user_text, system_text = _caption_prompt_parts(rec, config)
            if not user_text.strip():
                user_text = "Describe this clip."
            if system_text:
                batch.append((Image.new("RGB", (64, 64)), user_text, system_text))
            else:
                batch.append((Image.new("RGB", (64, 64)), user_text))
        else:
            user_text, system_text = _caption_prompt_parts(rec, config)
            if system_text:
                batch.append((images, user_text, system_text))
            else:
                batch.append((images, user_text))
        order.append(rec)

    try:
        vllm_cfg = config.get("models", {}).get("vllm", {})
        max_tokens = int(
            vllm_cfg.get("max_tokens", config.get("pipeline", {}).get("captioner", {}).get("max_tokens", 1000))
        )
        raws = engine.generate_batch(batch, max_tokens=max_tokens)
    finally:
        if own_engine:
            QwenVLLMEngine.release()

    results: List[str] = []
    for rec, raw in zip(order, raws):
        results.append(raw or "")
    return results
