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
            path = str(models_root(config) / "gemma-3-4b-it")
        self.model_path = str(path)
        self.display_name = gcfg.get("model_name", "Gemma-3-4B-IT (verify)")
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
                "Download e.g.: hf download google/gemma-3-4b-it --local-dir "
                f"{self.model_path}"
            )
        log_service_gpus("s6", "VLM verify — Gemma bucket check", self.model_path, self.gpu_ids)
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        try:
            import flash_attn  # noqa: F401
            attn = "flash_attention_2"
        except ImportError:
            attn = "sdpa"

        load_kwargs: Dict[str, Any] = {
            "dtype": torch.bfloat16,
            "attn_implementation": attn,
        }
        if len(self.gpu_ids) == 1:
            load_kwargs["device_map"] = f"cuda:{self.gpu_ids[0]}"
        else:
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = {i: "70GiB" for i in self.gpu_ids}

        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, **load_kwargs
        ).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_path)

    def verify(self, image: Image.Image, bucket: str) -> Dict[str, Any]:
        self.load()
        import torch

        prompt = (
            f"A classifier assigned this frame to bucket '{bucket}' "
            "(bucket_01 people_portraits, bucket_03 architecture, bucket_04 landscape, etc.). "
            "Does the frame content match that bucket?\n"
            "Return ONLY JSON:\n"
            '{"verified":true,"confidence":0.9,"route":"people|other","bucket_matches":true}\n'
            "Set verified=false and bucket_matches=false if the assigned bucket is wrong. "
            "route=people only for clear human/portrait focus; otherwise route=other."
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
        data = parse_vlm_json(
            raw,
            {"verified": False, "confidence": 0.5, "route": "other", "bucket_matches": False},
        )
        if "bucket_matches" in data:
            data["verified"] = bool(data["bucket_matches"])
        return data

    def cleanup(self) -> None:
        import gc

        import torch

        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
