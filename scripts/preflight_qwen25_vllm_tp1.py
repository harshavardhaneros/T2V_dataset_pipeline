#!/usr/bin/env python3
import os, sys
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
from vllm import LLM
MODEL = "/mnt/data0/harsha/new_dataset_pipeline/models/Qwen2.5-VL-7B-Instruct"
llm = LLM(model=MODEL, tensor_parallel_size=1, dtype="bfloat16", trust_remote_code=True,
          max_model_len=4096, limit_mm_per_prompt={"image": 1}, enforce_eager=True)
print("SUCCESS TP=1")
