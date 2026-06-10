#!/usr/bin/env bash
# Install vLLM + PyTorch matched to host NVIDIA driver (cu124).
# Do NOT pip install -U vllm without pins — vLLM 0.22+ upgrades torch to cu130
# which breaks CUDA on driver 550 / CUDA 12.4 hosts.
set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate indic_video_pipeline

echo "=== Driver ==="
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null | head -1 || true

echo "=== Remove broken flash-attn (incompatible with torch cu124) ==="
pip uninstall -y flash-attn flash_attn 2>/dev/null || true

echo "=== PyTorch cu124 ==="
pip install --force-reinstall \
  "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
  --index-url https://download.pytorch.org/whl/cu124

python -c "
import torch
assert torch.cuda.is_available(), 'CUDA unavailable after torch install'
print('torch', torch.__version__, 'devices:', torch.cuda.device_count())
"

echo "=== vLLM 0.8.5 (cu124-compatible) ==="
pip install "vllm==0.8.5"

echo "=== Pin transformers (avoid torch 2.7+ requirement from v5.x) ==="
pip install "transformers>=4.51.0,<4.52.0" "tokenizers>=0.21,<0.22"

echo "=== Re-pin torch if vllm upgraded it ==="
pip install --force-reinstall \
  "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
  --index-url https://download.pytorch.org/whl/cu124

echo "=== Verify ==="
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
python -c "
import os
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
import torch, vllm
assert torch.cuda.is_available()
print('torch', torch.__version__)
print('vllm', vllm.__version__)
print('CUDA OK')
"

echo "Done. Configs: pipeline_v3_vllm_2gpu.yaml | pipeline_v3_vllm_8gpu.yaml"
