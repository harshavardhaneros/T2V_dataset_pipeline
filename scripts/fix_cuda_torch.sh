#!/usr/bin/env bash
# Fix indic_video_pipeline PyTorch to match driver (cu128).
set -euo pipefail
conda run -n indic_video_pipeline pip install \
  "torch==2.10.0" torchvision \
  --index-url https://download.pytorch.org/whl/cu128
conda run -n indic_video_pipeline python -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA still False'; print('CUDA OK', torch.cuda.device_count(), 'GPUs')"
