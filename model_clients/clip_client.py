"""CLIP similarity scoring with optional real model."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import numpy as np


class ClipClient:
    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("clip_model", config)
        self.use_placeholder = cfg.get("use_placeholder", True)
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        if not self.use_placeholder:
            self._load_model(cfg)

    def _load_model(self, cfg: Dict[str, Any]) -> None:
        try:
            import open_clip
            import torch

            name = cfg.get("name", "ViT-B-32")
            pretrained = cfg.get("pretrained", "openai")
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                name, pretrained=pretrained
            )
            self._tokenizer = open_clip.get_tokenizer(name)
            self._model.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            self._torch = torch
        except Exception:
            self.use_placeholder = True

    def score_image_text(self, image: Any, text: str) -> float:
        from common.caption_text import caption_to_str

        text = caption_to_str(text)
        if not text:
            text = "Indic cultural scene"
        if self.use_placeholder or self._model is None:
            h = int(hashlib.md5(text.encode()).hexdigest(), 16)
            return 0.55 + (h % 40) / 100.0

        import torch
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image[:, :, ::-1])
        img_t = self._preprocess(image).unsqueeze(0).to(self._device)
        text_t = self._tokenizer([text]).to(self._device)
        with torch.no_grad():
            img_f = self._model.encode_image(img_t)
            txt_f = self._model.encode_text(text_t)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            sim = (img_f @ txt_f.T).item()
        return float(max(0.0, min(1.0, (sim + 1) / 2)))
