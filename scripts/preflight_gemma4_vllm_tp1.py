#!/usr/bin/env python3
import os, sys
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
from vllm import LLM
MODEL = "/mnt/data0/harsha/new_dataset_pipeline/models/gemma-4-31b-it"
print("loading gemma4 TP=1...")
llm = LLM(model=MODEL, tensor_parallel_size=1, dtype="bfloat16", trust_remote_code=True,
          max_model_len=4096, limit_mm_per_prompt={"image": 3}, enforce_eager=True)
print("SUCCESS gemma4 TP=1")
