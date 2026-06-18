#!/usr/bin/env bash
# vLLM + transformers stack for Gemma-4-31B-IT captioning (s8).
# Qwen2.5 s5 classify still works on the same env; for Qwen-only pin use install_vllm.sh.
set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate indic_video_pipeline

echo "=== Driver ==="
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null | head -1 || true

echo "=== vLLM 0.19+ (Gemma4 multimodal) ==="
pip install "vllm==0.19.0"

echo "=== transformers 5.x (Gemma4Processor) ==="
pip install "transformers>=5.5.0"

echo "=== Verify Gemma4 config loads ==="
python -c "
from transformers import AutoConfig, AutoProcessor
mp = '/mnt/data0/harsha/new_dataset_pipeline/models/gemma-4-31b-it'
cfg = AutoConfig.from_pretrained(mp, trust_remote_code=True)
assert cfg.model_type == 'gemma4', cfg.model_type
proc = AutoProcessor.from_pretrained(mp)
print('gemma4 processor', type(proc).__name__)
import vllm
print('vllm', vllm.__version__)
"

echo "Done. Set captioner.caption_model: gemma4 and backend: vllm in pipeline_v3_vllm_*.yaml"
