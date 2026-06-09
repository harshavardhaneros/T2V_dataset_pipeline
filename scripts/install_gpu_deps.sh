#!/usr/bin/env bash
# flash-attn + onnxruntime-gpu for Qwen-VL and InsightFace CUDA
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate indic_video_pipeline

echo "Installing onnxruntime-gpu (InsightFace CUDA)..."
pip install -U onnxruntime-gpu

echo "Installing flash-attn (Qwen-VL — may take 10+ min)..."
pip install flash-attn --no-build-isolation

python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'n_gpu', torch.cuda.device_count())
try:
    import flash_attn
    print('flash_attn OK')
except ImportError:
    print('flash_attn MISSING')
try:
    import onnxruntime as ort
    print('onnxruntime providers:', ort.get_available_providers())
except Exception as e:
    print('onnxruntime', e)
"
