#!/usr/bin/env python3
"""Smoke test data-parallel vLLM (TP=1 per GPU) on 2 logical GPUs."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image

from common.qwen_vllm import QwenVLLMEngine
from run_pipeline import load_config, pipeline_root


def _mem_gb() -> str:
    try:
        import torch

        parts = []
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            parts.append(f"cuda:{i} {free/1e9:.1f}/{total/1e9:.1f}GB free")
        return " | ".join(parts) or "no cuda"
    except Exception as exc:
        return f"mem query failed: {exc}"


def _run_stage(config: dict, stage: str, n_items: int = 4) -> None:
    print(f"\n=== {stage} data-parallel smoke ({n_items} prompts) ===")
    print("GPUs visible:", os.environ.get("CUDA_VISIBLE_DEVICES", "all"))
    print("Before load:", _mem_gb())
    t0 = time.time()
    engine = QwenVLLMEngine.acquire(config, stage=stage)
    print(
        f"Engine groups: {engine.gpu_groups} "
        f"(TP={getattr(engine, 'tensor_parallel_size', '?')})"
    )
    img = Image.new("RGB", (224, 224), color="blue")
    items = [(img, f"Describe this image in one short sentence. item={i}.") for i in range(n_items)]
    texts = engine.generate_chunks(items, batch_size=2, progress_desc=f"{stage} dp")
    QwenVLLMEngine.release()
    print(f"After run: {_mem_gb()}")
    print(f"OK {stage} in {time.time()-t0:.1f}s — sample: {texts[0][:120]!r}")


def main() -> int:
    cfg = load_config(pipeline_root(), "pipeline_v3_vllm_2gpu.yaml")
    if len(sys.argv) > 1 and sys.argv[1] == "s8-only":
        _run_stage(cfg, "s8", n_items=2)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "s5-only":
        _run_stage(cfg, "s5", n_items=4)
        return 0
    _run_stage(cfg, "s5", n_items=4)
    _run_stage(cfg, "s8", n_items=2)
    print("\nALL DP SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
