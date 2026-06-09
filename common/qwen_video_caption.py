"""Qwen2.5-VL native video clip captioning (MP4 in → caption out)."""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.actor_caption import enforce_actor_names_in_caption
from common.clip_io import export_clip_mp4, frame_offsets_for_record
from common.gemma_caption import (
    CAPTION_SYSTEM_PROMPT,
    build_caption_user_text,
    parse_caption_json,
    to_single_line_json,
)
from common.gpu_info import log_service_gpus, resolve_gpu_ids
from common.paths import qwen_video_model_path
from common.screen_position import known_actor_names

logger = logging.getLogger(__name__)


def ensure_clip_mp4(
    movie_video: Path,
    record: Dict[str, Any],
    clips_dir: Path,
    config: Dict[str, Any],
) -> Optional[Path]:
    """Export a per-clip MP4 if missing (used as Qwen video input)."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    out = clips_dir / f"{record['clip_id']}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out
    export_cfg = config.get("pipeline", {}).get("export", {})
    thresholds = config.get("thresholds", {})
    if export_clip_mp4(
        movie_video,
        record,
        out,
        export_cfg=export_cfg,
        thresholds=thresholds,
    ):
        return out
    return None


def build_video_caption_prompt(rec: Dict[str, Any], config: Dict[str, Any]) -> str:
    offsets = frame_offsets_for_record(rec, config)
    user = build_caption_user_text(
        rec, multi_frame=True, frame_offsets=offsets
    )
    return f"{CAPTION_SYSTEM_PROMPT}\n\n{user}"


class QwenVideoCaptionWorker:
    """Single-GPU Qwen2.5-VL video captioner (used standalone or in Ray actors)."""

    def __init__(self, config: Dict[str, Any], device: str = "cuda:0"):
        self._config = config
        qc = config.get("models", {}).get("qwen_video_caption", {})
        pcfg = config.get("pipeline", {}).get("captioner", {})
        self.model_path = str(qwen_video_model_path(config))
        self.device = device
        self.fps = float(qc.get("video_fps", pcfg.get("video_fps", 1.0)))
        self.max_pixels = int(qc.get("max_pixels", pcfg.get("max_pixels", 360 * 420)))
        self.max_new_tokens = int(qc.get("max_tokens", pcfg.get("max_tokens", 800)))
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        model_dir = Path(self.model_path)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"Qwen video model not found: {model_dir}\n"
                "Run: bash scripts/download_qwen_vl_7b.sh"
            )

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"

        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation=attn_impl,
        ).eval()

    def caption_video(self, clip_path: Path, prompt: str) -> str:
        from qwen_vl_utils import process_vision_info

        self.load()
        video_uri = f"file://{clip_path.resolve()}"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_uri,
                        "fps": self.fps,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            fps=self.fps,
            **video_kwargs,
        ).to(self._model.device)

        import torch

        with torch.no_grad():
            gen_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = gen_ids[:, inputs.input_ids.shape[1] :]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def cleanup(self) -> None:
        import torch

        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class QwenVideoCaptionService:
    """Qwen2.5-VL video captioning — single GPU or Ray multi-GPU pool."""

    _shared: Optional["QwenVideoCaptionService"] = None

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        qc = config.get("models", {}).get("qwen_video_caption", {})
        pcfg = config.get("pipeline", {}).get("captioner", {})
        self.model_path = str(qwen_video_model_path(config))
        self.gpu_ids = resolve_gpu_ids(
            [int(g) for g in qc.get("gpu_ids", pcfg.get("gpu_ids", [0]))]
        )
        self.fps = float(qc.get("video_fps", pcfg.get("video_fps", 1.0)))
        self.max_new_tokens = int(qc.get("max_tokens", pcfg.get("max_tokens", 800)))
        self._worker: Optional[QwenVideoCaptionWorker] = None

    @classmethod
    def acquire(cls, config: Dict[str, Any]) -> "QwenVideoCaptionService":
        if cls._shared is None:
            cls._shared = cls(config)
        return cls._shared

    @classmethod
    def release(cls) -> None:
        if cls._shared:
            cls._shared.cleanup()
        cls._shared = None

    def _use_ray_gpus(self) -> bool:
        rc = self._config.get("pipeline", {}).get("ray", {})
        return bool(rc.get("parallel_gpu_caption", False)) and len(self.gpu_ids) > 1

    def caption_records(
        self,
        items: List[Tuple[Dict[str, Any], Path]],
    ) -> List[str]:
        if not items:
            return []

        log_service_gpus(
            "s8",
            f"Qwen2.5-VL video caption @ {self.fps} fps",
            self.model_path,
            self.gpu_ids,
            extra="Ray multi-GPU" if self._use_ray_gpus() else "single GPU",
        )

        if self._use_ray_gpus():
            return self._caption_records_ray(items)
        return self._caption_records_single(items)

    def _caption_records_single(
        self, items: List[Tuple[Dict[str, Any], Path]]
    ) -> List[str]:
        if self._worker is None:
            gpu = self.gpu_ids[0] if self.gpu_ids else 0
            self._worker = QwenVideoCaptionWorker(
                self._config, device=f"cuda:{gpu}"
            )

        results: List[str] = []
        for rec, clip_path in items:
            try:
                prompt = build_video_caption_prompt(rec, self._config)
                raw = self._worker.caption_video(clip_path, prompt)
                actors = rec.get("clip_actors") or known_actor_names(
                    rec.get("actors") or []
                )
                if actors:
                    struct = parse_caption_json(raw)
                    short = struct.get("short_description", "")
                    if short:
                        struct["short_description"] = enforce_actor_names_in_caption(
                            short, actors
                        )
                        raw = to_single_line_json(
                            json.dumps(struct, ensure_ascii=False)
                        )
                results.append(raw)
            except Exception as exc:
                logger.warning("Video caption failed for %s: %s", rec.get("clip_id"), exc)
                results.append("")
        return results

    def _caption_records_ray(self, items: List[Tuple[Dict[str, Any], Path]]) -> List[str]:
        from common.gpu_actor_pool import gpu_actor_count
        from common.ray_pool import init_ray
        from common.vlm_ray_actors import QwenVideoCaptionActor

        if QwenVideoCaptionActor is None or not init_ray(self._config):
            logger.warning("Ray GPU caption unavailable; falling back to single GPU")
            return self._caption_records_single(items)

        payloads = [
            {
                "record": rec,
                "clip_path": str(clip_path),
                "config": self._config,
            }
            for rec, clip_path in items
        ]
        n_actors = gpu_actor_count(self._config, self.gpu_ids)
        import ray
        from common.ray_pool import init_ray as _init

        _init(self._config)
        actors = [QwenVideoCaptionActor.remote(self._config) for _ in range(n_actors)]
        futures = [
            actors[i % n_actors].caption.remote(payload)
            for i, payload in enumerate(payloads)
        ]
        try:
            rows = ray.get(futures)
        finally:
            for actor in actors:
                try:
                    ray.kill(actor)
                except Exception:
                    pass

        by_id = {row["clip_id"]: row.get("raw", "") for row in rows}
        return [by_id.get(rec["clip_id"], "") for rec, _ in items]

    def cleanup(self) -> None:
        if self._worker:
            self._worker.cleanup()
            self._worker = None
        gc.collect()
