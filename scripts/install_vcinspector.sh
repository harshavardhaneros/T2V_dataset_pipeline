#!/usr/bin/env bash
# Isolated conda env for VCInspector (ms-swift VllmEngine + dipta007/VCInspector-7B).
# Keeps vLLM/swift versions separate from indic_video_pipeline (gemma4 caption).
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx vcinspector; then
  echo "conda env vcinspector already exists"
else
  conda create -y -n vcinspector python=3.10
fi

conda activate vcinspector

echo "=== PyTorch cu124 ==="
pip install --force-reinstall \
  "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
  --index-url https://download.pytorch.org/whl/cu124

echo "=== ms-swift + vLLM + qwen-vl-utils ==="
pip install "ms-swift" "vllm>=0.8.5" "qwen-vl-utils" "transformers>=4.51.0" "accelerate"

echo "=== Download VCInspector-7B ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download('dipta007/VCInspector-7B', local_dir_use_symlinks=True)
print('model cached')
"

echo "=== Verify swift VllmEngine ==="
python -c "from swift.infer_engine import VllmEngine, InferRequest, RequestConfig; print('swift OK')"

echo ""
echo "Done. Run s13 via main pipeline (indic_video_pipeline env) — worker uses: conda run -n vcinspector"
