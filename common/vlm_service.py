"""Shared Qwen-VL-32B loader for classify + caption (Master vlm_backend)."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from common.gpu_info import log_service_gpus, resolve_gpu_ids
from common.master_bridge import master_import_context
from common.paths import qwen_model_path
from common.video_time import clip_local_range


def parse_vlm_json(raw: str, fallback: Optional[dict] = None) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return fallback or {"_parse_error": True}


def clip_keyframe_images(
    video_path: Path,
    record: Dict[str, Any],
    config: Dict[str, Any],
    fractions: Optional[List[float]] = None,
) -> List[Image.Image]:
    from common.frame_sampler import sample_keyframes

    if fractions is None:
        fractions = [0.25, 0.5, 0.75]
    start, end = clip_local_range(record, config)
    frames = sample_keyframes(
        str(video_path),
        start,
        end,
        fractions=fractions,
        crop_box=record.get("crop_box", ""),
    )
    return [Image.fromarray(bgr[:, :, ::-1]) for bgr in frames]


def majority_vote_buckets(votes: List[str]) -> Tuple[str, float]:
    if not votes:
        return "bucket_01", 0.0
    counter = Counter(votes)
    winner, count = counter.most_common(1)[0]
    return winner, count / len(votes)


class QwenVLMService:
    _shared: Optional["QwenVLMService"] = None

    def __init__(self, config: Dict[str, Any], gpu_ids: List[int], stage: str):
        mp = config["pipeline"]["master_pipeline"]
        self.config = config
        self.model_path = str(qwen_model_path(config))
        self.display_name = mp.get("vlm_model_name", "Qwen3-VL-32B")
        self.gpu_ids = resolve_gpu_ids(gpu_ids)
        self.stage = stage
        self.backend = None

    @classmethod
    def acquire(cls, config: Dict[str, Any], gpu_ids: List[int], stage: str) -> "QwenVLMService":
        if cls._shared is None:
            cls._shared = cls(config, gpu_ids, stage)
        return cls._shared

    def load(self) -> None:
        if self.backend is not None:
            return
        log_service_gpus(
            self.stage,
            f"VLM classify/caption — {self.display_name}",
            self.model_path,
            self.gpu_ids,
        )
        with master_import_context():
            from vlm_backend import create_backend

            mp = self.config["pipeline"]["master_pipeline"]
            self.backend = create_backend(
                mp.get("caption_backend", "transformers"),
                model_path=self.model_path,
                gpu_ids=self.gpu_ids,
                max_new_tokens=int(mp.get("max_new_tokens", 512)),
            )
            self.backend.load()

    def generate(self, image: Image.Image, prompt: str) -> str:
        self.load()
        return self.backend.generate(image, prompt)

    def generate_multi(self, images: List[Image.Image], prompt: str) -> str:
        """Multi-frame clip caption (Qwen-VL)."""
        self.load()
        if len(images) == 1:
            return self.generate(images[0], prompt)
        import torch
        from qwen_vl_utils import process_vision_info

        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        proc = self.backend.processor
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = proc(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        ).to(self.backend.model.device)
        with torch.no_grad():
            gen_ids = self.backend.model.generate(
                **inputs, max_new_tokens=self.backend.max_new_tokens, do_sample=False
            )
        trimmed = gen_ids[:, inputs.input_ids.shape[1] :]
        return proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    @classmethod
    def release(cls) -> None:
        if cls._shared and cls._shared.backend:
            try:
                cls._shared.backend.cleanup()
            except Exception:
                pass
        cls._shared = None
