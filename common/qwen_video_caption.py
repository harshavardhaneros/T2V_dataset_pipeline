"""Native MP4 clip captioning (Qwen-VL, Qwen3.5, Gemma4 dense)."""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.clip_io import export_clip_mp4, frame_offsets_for_record
from common.gemma_caption import (
    build_caption_user_text,
    get_caption_system_prompt,
)
from common.gpu_info import log_service_gpus, resolve_gpu_ids
from common.paths import video_caption_model_path

logger = logging.getLogger(__name__)


def ensure_clip_mp4(
    movie_video: Path,
    record: Dict[str, Any],
    clips_dir: Path,
    config: Dict[str, Any],
) -> Optional[Path]:
    """Export a per-clip MP4 if missing (used as video caption input)."""
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
    return f"{get_caption_system_prompt(config)}\n\n{user}"


def _read_model_type(model_path: Path) -> str:
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        return ""
    with cfg_path.open(encoding="utf-8") as fh:
        cfg = json.load(fh)
    return str(cfg.get("model_type", "")).lower()


class VideoCaptionWorker:
    """Single-GPU native video clip captioner (Qwen-VL / Qwen3.5 / Gemma4)."""

    def __init__(self, config: Dict[str, Any], device: str = "cuda:0"):
        self._config = config
        vc = config.get("models", {}).get("video_caption", {})
        qc = config.get("models", {}).get("qwen_video_caption", {})
        pcfg = config.get("pipeline", {}).get("captioner", {})
        self.model_path = str(video_caption_model_path(config))
        self.device = device
        self.fps = float(
            vc.get("video_fps", qc.get("video_fps", pcfg.get("video_fps", 1.0)))
        )
        self.max_pixels = int(
            vc.get("max_pixels", qc.get("max_pixels", pcfg.get("max_pixels", 360 * 420)))
        )
        self.max_new_tokens = int(
            vc.get("max_tokens", qc.get("max_tokens", pcfg.get("max_tokens", 800)))
        )
        self._model = None
        self._processor = None
        self._backend_kind = ""

    def load(self) -> None:
        if self._model is not None:
            return
        model_dir = Path(self.model_path)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"Video caption model not found: {model_dir}\n"
                "Run: bash scripts/download_caption_models.sh gemma4_dense qwen3.5"
            )

        import torch

        from common.attn_backend import resolve_attn_implementation

        model_type = _read_model_type(model_dir)
        attn_impl = resolve_attn_implementation()

        if model_type == "gemma4":
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            self._backend_kind = "gemma4"
            self._processor = AutoProcessor.from_pretrained(self.model_path)
            self._model = AutoModelForMultimodalLM.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation=attn_impl,
            ).eval()
            return

        if model_type.startswith("qwen3_5"):
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

            self._backend_kind = "qwen35"
            self._processor = AutoProcessor.from_pretrained(self.model_path)
            self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation=attn_impl,
            ).eval()
            return

        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._backend_kind = "qwen_vl"
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation=attn_impl,
        ).eval()

    def caption_video(self, clip_path: Path, prompt: str) -> str:
        self.load()
        if self._backend_kind == "gemma4":
            return self._caption_gemma4(clip_path, prompt)
        if self._backend_kind == "qwen35":
            return self._caption_qwen35(clip_path, prompt)
        return self._caption_qwen_family(clip_path, prompt)

    def _caption_qwen35(self, clip_path: Path, prompt: str) -> str:
        import torch

        video_path = str(clip_path.resolve())
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "fps": self.fps},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            fps=self.fps,
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        return self._processor.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

    def _caption_gemma4(self, clip_path: Path, prompt: str) -> str:
        import torch

        video_path = str(clip_path.resolve())
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        response = self._processor.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        return response.strip()

    def _caption_qwen_family(self, clip_path: Path, prompt: str) -> str:
        from qwen_vl_utils import process_vision_info

        import torch

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
            **{k: v for k, v in video_kwargs.items() if k != "fps"},
        ).to(self._model.device)

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


class QwenVideoCaptionWorker(VideoCaptionWorker):
    """Back-compat alias."""


class VideoCaptionService:
    """Native MP4 video captioning — single GPU or Ray multi-GPU pool."""

    _shared: Optional["VideoCaptionService"] = None

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        vc = config.get("models", {}).get("video_caption", {})
        qc = config.get("models", {}).get("qwen_video_caption", {})
        pcfg = config.get("pipeline", {}).get("captioner", {})
        self.model_path = str(video_caption_model_path(config))
        self.gpu_ids = resolve_gpu_ids(
            [int(g) for g in vc.get("gpu_ids", qc.get("gpu_ids", pcfg.get("gpu_ids", [0])))]
        )
        self.fps = float(
            vc.get("video_fps", qc.get("video_fps", pcfg.get("video_fps", 1.0)))
        )
        self.max_new_tokens = int(
            vc.get("max_tokens", qc.get("max_tokens", pcfg.get("max_tokens", 800)))
        )
        self._worker: Optional[VideoCaptionWorker] = None

    @classmethod
    def acquire(cls, config: Dict[str, Any]) -> "VideoCaptionService":
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
            f"Native MP4 video caption @ {self.fps} fps",
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
            self._worker = VideoCaptionWorker(self._config, device=f"cuda:{gpu}")

        results: List[str] = []
        for rec, clip_path in items:
            try:
                prompt = build_video_caption_prompt(rec, self._config)
                raw = self._worker.caption_video(clip_path, prompt)
                results.append(raw)
            except Exception as exc:
                logger.warning(
                    "Video caption failed for %s: %s", rec.get("clip_id"), exc
                )
                results.append("")
        return results

    def _caption_records_ray(self, items: List[Tuple[Dict[str, Any], Path]]) -> List[str]:
        from common.gpu_actor_pool import gpu_actor_count
        from common.ray_pool import init_ray
        from common.vlm_ray_actors import VideoCaptionActor

        if VideoCaptionActor is None or not init_ray(self._config):
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

        actors = [VideoCaptionActor.remote(self._config) for _ in range(n_actors)]
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


class QwenVideoCaptionService(VideoCaptionService):
    """Back-compat alias."""
