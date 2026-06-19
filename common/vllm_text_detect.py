"""vLLM batched on-screen text detection for s4 (Gemma4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

from common.qwen_vllm import QwenVLLMEngine
from common.vlm_service import parse_vlm_json

TEXT_DETECT_PROMPT = (
    "Does this video frame contain visible readable text?\n"
    "Include: subtitles, credits, titles, captions, chyrons, logos with words, "
    "any overlaid or burned-in text.\n"
    "Do NOT count pure scenery, faces, or costumes with no readable words.\n"
    "Return ONLY JSON:\n"
    '{"has_text": true, "text_type": "credits|subtitles|title|logo_text|other|none", '
    '"confidence": 0.9, "summary": "brief note"}\n'
    "Set has_text=false and text_type=none when no readable text is visible."
)


def _frame_to_pil(frame) -> Image.Image:
    import cv2

    if frame is None or frame.size == 0:
        return Image.new("RGB", (64, 64), color=(0, 0, 0))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _load_middle_frame(
    rec: Dict[str, Any],
    movie_dir: Path,
    movie_video: Path | None,
) -> Image.Image:
    from common.frame_sampler import read_middle_frame, sample_keyframes
    from common.review_clips import find_workspace_clip

    clip_id = rec.get("clip_id", "")
    clip_path = find_workspace_clip(movie_dir, clip_id) if clip_id else None
    if clip_path and clip_path.exists():
        import cv2

        cap = cv2.VideoCapture(str(clip_path))
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            mid = int(total / 2) if total > 0 else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ok, frame = cap.read()
            cap.release()
            if ok:
                return _frame_to_pil(frame)

    if movie_video and movie_video.exists():
        clip_mp4 = movie_dir / "clips" / f"{clip_id}.mp4"
        if clip_mp4.exists():
            frames = sample_keyframes(
                str(clip_mp4), 0.0, float(rec.get("duration", 5)), fractions=[0.5]
            )
            if frames:
                return _frame_to_pil(frames[0])
        frame = read_middle_frame(
            str(movie_video),
            rec["timestamp_start"],
            rec["timestamp_end"],
            crop_box=rec.get("crop_box", ""),
        )
        if frame is not None:
            return _frame_to_pil(frame)

    return Image.new("RGB", (64, 64), color=(0, 0, 0))


def detect_text_clips_vllm(
    config: Dict[str, Any],
    targets: List[Dict[str, Any]],
    *,
    movie_dir: Path,
    movie_video: Path | None,
) -> Dict[str, Dict[str, Any]]:
    """Run Gemma4 vLLM text detection on middle frame per clip."""
    s4_cfg = config.get("pipeline", {}).get("s4", {})
    max_tokens = int(s4_cfg.get("max_tokens", 128))

    batch_items: List[Tuple[Image.Image, str]] = []
    clip_ids: List[str] = []
    for rec in targets:
        clip_ids.append(rec["clip_id"])
        image = _load_middle_frame(rec, movie_dir, movie_video)
        batch_items.append((image, TEXT_DETECT_PROMPT))

    engine = QwenVLLMEngine.acquire(config, stage="s4")
    try:
        raws = engine.generate_chunks(
            batch_items,
            max_tokens=max_tokens,
            progress_desc="s4 text detect",
        )
    finally:
        QwenVLLMEngine.release()

    results: Dict[str, Dict[str, Any]] = {}
    for clip_id, raw in zip(clip_ids, raws):
        data = parse_vlm_json(
            raw,
            {"has_text": False, "text_type": "none", "confidence": 0.0, "summary": ""},
        )
        has_text = bool(data.get("has_text"))
        text_type = str(data.get("text_type") or ("other" if has_text else "none"))
        confidence = float(data.get("confidence", 0.5 if has_text else 0.0))
        results[clip_id] = {
            "has_text": has_text,
            "text_overlay": {
                "present": has_text,
                "text_type": text_type,
                "confidence": round(confidence, 4),
                "summary": str(data.get("summary") or "")[:500],
                "model": "Gemma-4-31B-IT",
                "backend": "vllm",
                "raw": raw[:1000],
            },
        }
    return results
