#!/usr/bin/env python3
"""
VLM Backend abstraction for classification and captioning.

Supports two backends:
  - transformers: HuggingFace transformers (default, works everywhere)
  - vllm: vLLM for high-throughput batched inference (optional, 10-15x faster)

Usage:
    backend = create_backend("vllm", model_path="/path/to/model", gpu_ids=[0,1])
    backend.load()
    result = backend.generate(pil_image, prompt)
    results = backend.generate_batch([(img1, prompt1), (img2, prompt2)])
    backend.cleanup()
"""

import gc
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

from common import MODEL_ID

logger = logging.getLogger(__name__)


class VLMBackend(ABC):
    """Base class for VLM inference backends."""

    def __init__(self, model_path: str | None = None, gpu_ids: list[int] | None = None,
                 max_new_tokens: int = 512):
        self.model_path = model_path or MODEL_ID
        self.gpu_ids = gpu_ids or [0]
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def load(self):
        """Load model into memory."""

    @abstractmethod
    def generate(self, image: Image.Image, prompt: str) -> str:
        """Generate text from a single image + prompt."""

    def generate_batch(self, items: list[tuple[Image.Image, str]]) -> list[str]:
        """Generate text for a batch of (image, prompt) pairs.

        Default implementation processes sequentially; vLLM overrides with
        true batched inference.
        """
        return [self.generate(img, prompt) for img, prompt in items]

    @abstractmethod
    def cleanup(self):
        """Release GPU memory."""


class TransformersBackend(VLMBackend):
    """HuggingFace transformers backend (default)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.processor = None

    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        path = self.model_path
        logger.info(f"[transformers] Loading {path} on GPUs {self.gpu_ids}...")
        t0 = time.time()

        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        valid = [i for i in self.gpu_ids if i < n]
        if not valid:
            valid = list(range(n)) if n else [0]
        max_memory = {i: "75GiB" for i in valid}
        self.gpu_ids = valid

        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        except ImportError:
            attn_impl = "sdpa"
            # SDPA has a confirmed precision bug with Qwen2.5-VL: ViT features
            # can be ~30x larger than text embeddings, causing numerical instability.
            # Install flash-attn for better accuracy and memory efficiency:
            #   pip install flash-attn --no-build-isolation
            logger.warning(
                "flash-attn not installed — falling back to SDPA. "
                "SDPA has a known precision bug with Qwen2.5-VL (ViT/text embedding mismatch). "
                "Install flash-attn for correct results: pip install flash-attn --no-build-isolation"
            )

        self.model = AutoModelForImageTextToText.from_pretrained(
            path,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            attn_implementation=attn_impl,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(path)
        logger.info(f"Model loaded in {time.time() - t0:.1f}s ({type(self.model).__name__})")

    def generate(self, image: Image.Image, prompt: str) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        # Set explicit pixel bounds to prevent double-resize bug in qwen_vl_utils
        # and cap memory use. 256*28*28 = min, 1280*28*28 = max (~1024px equivalent).
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        ).to(self.model.device)

        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

        trimmed = gen_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def cleanup(self):
        import torch
        del self.model, self.processor
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class VLLMBackend(VLMBackend):
    """vLLM backend for high-throughput batched inference.

    Requires vllm>=0.11.0 for Qwen3-VL support.
    Provides 10-15x throughput improvement over transformers for batches of 16+.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm = None

    def load(self):
        from vllm import LLM

        path = self.model_path
        tp_size = len(self.gpu_ids)
        logger.info(f"[vLLM] Loading {path} with tensor_parallel_size={tp_size}...")
        t0 = time.time()

        self.llm = LLM(
            model=path,
            tensor_parallel_size=tp_size,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 1},
            # Qwen2.5-VL does not support torch.compile (vLLM V1 default).
            # enforce_eager=True disables it to prevent crash on load.
            enforce_eager=True,
        )
        logger.info(f"vLLM model loaded in {time.time() - t0:.1f}s")

    def generate(self, image: Image.Image, prompt: str) -> str:
        results = self.generate_batch([(image, prompt)])
        return results[0]

    def generate_batch(self, items: list[tuple[Image.Image, str]]) -> list[str]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=self.max_new_tokens,
        )

        prompts = []
        for image, prompt_text in items:
            prompts.append({
                "prompt": f"<|im_start|>user\n<image>\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"image": image},
            })

        outputs = self.llm.generate(prompts, sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]

    def cleanup(self):
        import torch
        del self.llm
        self.llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_backend(backend_type: str = "transformers", model_path: str | None = None,
                   gpu_ids: list[int] | None = None, max_new_tokens: int = 512) -> VLMBackend:
    """Factory function to create a VLM backend.

    Args:
        backend_type: "transformers" or "vllm"
        model_path: Path to model (local or HuggingFace ID)
        gpu_ids: List of GPU IDs to use
        max_new_tokens: Maximum tokens to generate

    Returns:
        VLMBackend instance (not yet loaded — call .load())
    """
    kwargs = dict(model_path=model_path, gpu_ids=gpu_ids, max_new_tokens=max_new_tokens)

    if backend_type == "vllm":
        try:
            import vllm  # noqa: F401
        except ImportError:
            logger.warning("vllm not installed — falling back to transformers backend. "
                           "Install with: pip install vllm")
            return TransformersBackend(**kwargs)
        return VLLMBackend(**kwargs)

    return TransformersBackend(**kwargs)
