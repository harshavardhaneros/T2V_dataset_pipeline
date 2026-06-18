#!/usr/bin/env python3
import os
import sys

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM

MODEL = "/mnt/data0/harsha/new_dataset_pipeline/models/Qwen2.5-VL-7B-Instruct"


def main() -> int:
    print("loading qwen2.5 with vllm...")
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
        distributed_executor_backend="mp",
    )
    print("SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
