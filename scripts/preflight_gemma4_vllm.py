#!/usr/bin/env python3
"""Quick Gemma4 + vLLM load/infer smoke test (run with CUDA_VISIBLE_DEVICES set)."""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

MODEL = "/mnt/data0/harsha/new_dataset_pipeline/models/gemma-4-31b-it"


def main() -> int:
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL)
    print(f"processor loaded ({time.time() - t0:.1f}s)")

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
        distributed_executor_backend="mp",
    )
    print(f"llm loaded ({time.time() - t0:.1f}s)")

    img = Image.new("RGB", (224, 224), color="red")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        }
    ]
    prompt = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    out = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": img}}],
        SamplingParams(temperature=0, max_tokens=48),
    )
    text = out[0].outputs[0].text.strip()
    print(f"SUCCESS ({time.time() - t0:.1f}s): {text[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
