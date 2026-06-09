# Ready-to-run checklist

## Completed on this machine

| Item | Status |
|------|--------|
| Conda env `indic_video_pipeline` | Created |
| PyTorch CUDA (cu128) | `torch.cuda.is_available() == True`, 8× H200 |
| Qwen2.5-VL-32B-Instruct | Downloaded to `models/Qwen2.5-VL-32B-Instruct` (~64GB) |
| YOLO face weights | `Master_Pipeline_t2i_dataset/actors/yolov12n-face.pt` |
| Actor embeddings | 108 `.pkl` files |
| Config `model_path` | Points to local `models/` folder |

## Before you run

```bash
conda activate indic_video_pipeline
cd /mnt/data0/harsha/new_dataset_pipeline/Indic_video_pipeline
bash scripts/verify_setup.sh
```

## Test run (30 min: minutes 100–130)

Fast validation before the full movie (~15–25 min total for phased test).

```bash
conda activate indic_video_pipeline
cd /mnt/data0/harsha/new_dataset_pipeline/Indic_video_pipeline

# One-shot test script (segment + s1–s7 + inspect + s8 on 15 clips + s9–s12)
bash scripts/run_test_pipeline.sh
```

Or step by step:

```bash
# 1) Cut segment (stream copy, ~1 min)
bash scripts/make_test_segment.sh /mnt/data0/harsha/Movies/feb_11/ABCD.mp4 100 130

# 2) Phase A: full test through actor tagging (~5–10 min)
python run_pipeline.py \
  --movie test_segments/ABCD_min100-130.mp4 \
  --video-id ABCD_test_100_130 \
  --time-offset 6000 \
  --force \
  --to-step s7

python scripts/inspect_test_output.py --workspace workspaces/ABCD_test_100_130

# 3) Phase B: caption only, 15 clips with real Qwen (~10–20 min)
python run_pipeline.py \
  --movie test_segments/ABCD_min100-130.mp4 \
  --video-id ABCD_test_100_130 \
  --time-offset 6000 \
  --from-step s8 \
  --to-step s12 \
  --max-clips 15 \
  --force

python scripts/inspect_test_output.py --workspace workspaces/ABCD_test_100_130 --show-captions
```

`--time-offset 6000` keeps timestamps aligned to the original ABCD.mp4 (minute 100 = 6000s).

Outputs: `workspaces/ABCD_test_100_130/` (separate from full `ABCD` run).

## Full pipeline (one command)

```bash
python run_pipeline.py \
  --movie /mnt/data0/harsha/Movies/feb_11/ABCD.mp4 \
  --video-id ABCD \
  --force
```

**GPU usage**

- **s7** actor tagging → GPU **4** (`actor_tag_gpu_id`)
- **s8** captions → GPUs **0–3** (`caption_gpu_ids`, transformers multi-GPU load)

**Runtime (ABCD, approximate)**

- s1 extract: ~4 min
- s7 actor tag: ~1–2 min (568 people clips)
- s8 caption: **long** with transformers (~1587 clips) — consider running `--from-step s8` overnight, or install vLLM later for 10× speedup
- s9 quality: ~3–4 min

## Outputs

See `workspaces/ABCD/`, `logs/s1..s12/`, `reports/`.

## If CUDA breaks after pip installs

```bash
bash scripts/fix_cuda_torch.sh
```
