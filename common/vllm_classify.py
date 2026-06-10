"""vLLM batched bucket classification for s5."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from common.qwen_classify import CLASSIFY_JSON_PROMPT, _normalize_bucket
from common.qwen_vllm import QwenVLLMEngine
from common.vlm_service import parse_vlm_json


def classify_clips_vllm(
    config: Dict[str, Any],
    targets: List[Dict[str, Any]],
    valid_buckets: List[str],
    *,
    video_path: str,
    clips_dir: Path,
    fractions: List[float],
) -> Dict[str, Dict[str, Any]]:
    from common.frame_sampler import sample_keyframes
    from common.video_time import clip_local_range

    engine = QwenVLLMEngine.acquire(config, stage="s5")
    batch_items: List[tuple] = []
    clip_ids: List[str] = []

    for rec in targets:
        if not rec.get("keep", True):
            continue
        clip_id = rec["clip_id"]
        clip_ids.append(clip_id)

        clip_mp4 = clips_dir / f"{clip_id}.mp4"
        frames = []
        if clip_mp4.exists() and clip_mp4.stat().st_size > 0:
            frames = sample_keyframes(
                str(clip_mp4),
                0.0,
                float(rec.get("duration", 5)),
                fractions=fractions,
                crop_box="",
            )
        elif video_path:
            start, end = clip_local_range(rec, config)
            frames = sample_keyframes(
                video_path,
                start,
                end,
                fractions=fractions,
                crop_box=rec.get("crop_box", ""),
            )

        if not frames:
            batch_items.append((Image.new("RGB", (64, 64)), CLASSIFY_JSON_PROMPT))
            continue
        mid = frames[len(frames) // 2]
        batch_items.append((Image.fromarray(mid[:, :, ::-1]), CLASSIFY_JSON_PROMPT))

    try:
        raws = engine.generate_chunks(
            batch_items, max_tokens=128, progress_desc="s5 classify"
        )
    finally:
        QwenVLLMEngine.release()

    results: Dict[str, Dict[str, Any]] = {}
    for clip_id, raw in zip(clip_ids, raws):
        data = parse_vlm_json(raw, {})
        bucket = _normalize_bucket(data.get("bucket", ""), valid_buckets)
        reject = bool(data.get("reject", False))
        conf = float(data.get("bucket_confidence", 0.5))
        results[clip_id] = {
            "clip_id": clip_id,
            "bucket": bucket,
            "bucket_confidence": round(conf, 4),
            "reject": reject,
            "reject_reason": data.get("reject_reason") if reject else None,
        }
    return results
