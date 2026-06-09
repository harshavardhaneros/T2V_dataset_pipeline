"""Gemma VLM verifier (2nd pass) — separate from Qwen classify/caption."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from common.gpu_info import log_service_gpus, resolve_gpu_ids
from common.paths import models_root
from common.vlm_service import parse_vlm_json


class GemmaVerifyService:
    def __init__(self, config: Dict[str, Any]):
        gcfg = config.get("models", {}).get("gemma", {})
        pcfg = config["pipeline"].get("master_pipeline", {})
        path = gcfg.get("model_path") or pcfg.get("gemma_model_path")
        if not path:
            path = str(models_root(config) / "gemma-3-27b-it")
        self.model_path = str(path)
        self.display_name = gcfg.get("model_name", "Gemma-27B (verify)")
        self.gpu_ids = resolve_gpu_ids(
            [int(g) for g in gcfg.get("gpu_ids", pcfg.get("verify_gpu_ids", [5, 6]))]
        )
        self.max_new_tokens = int(gcfg.get("max_new_tokens", 256))
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not Path(self.model_path).joinpath("config.json").exists():
            raise FileNotFoundError(
                f"Gemma verify model not found: {self.model_path}\n"
                "Download e.g.: hf download google/gemma-3-27b-it --local-dir "
                f"{self.model_path}"
            )
        log_service_gpus("s6", "VLM verify — Gemma", self.model_path, self.gpu_ids)
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        try:
            import flash_attn  # noqa: F401
            attn = "flash_attention_2"
        except ImportError:
            attn = "sdpa"

        max_memory = {i: "70GiB" for i in self.gpu_ids}
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            attn_implementation=attn,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_path)

    def verify(self, image: Image.Image, bucket: str) -> Dict[str, Any]:
        self.load()
        import torch

        prompt = (
            f"Verify this image belongs to cultural bucket '{bucket}'. "
            "Return ONLY JSON: "
            '{"verified":true,"confidence":0.9,"route":"people|other"}'
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(self._model.device)
        except Exception:
            inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(
                self._model.device
            )

        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = out[:, inputs.input_ids.shape[1] :]
        raw = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return parse_vlm_json(
            raw,
            {"verified": True, "confidence": 0.5, "route": "other"},
        )

    def cleanup(self) -> None:
        import gc

        import torch

        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
