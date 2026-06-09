"""Service 8: Gemma-3-4B structured captions (eros) or legacy Qwen captioning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from common.actor_caption import enforce_actor_names_in_caption
from common.base_service import BaseService
from common.caption_text import caption_to_str
from common.clip_io import extract_clip_frames
from common.gemma_caption import (
    GemmaCaptionService,
    parse_caption_json,
    pick_caption_frame,
    pick_caption_frames,
    to_single_line_json,
)
from common.qwen_video_caption import (
    QwenVideoCaptionService,
    ensure_clip_mp4,
)
from common.master_bridge import (
    bucket_to_category,
    build_caption_prompt,
    init_master,
    load_master_prompts,
    save_clip_keyframe,
)
from common.metadata_manager import MetadataManager
from common.paths import qwen_model_path
from common.vlm_service import QwenVLMService, clip_keyframe_images, parse_vlm_json

logger = logging.getLogger(__name__)


class CaptionService(BaseService):
    service_id = "s8"
    service_name = "s8_caption"
    owned_fields = ["caption", "caption_struct", "generated_caption", "prompt_version"]

    def _backend(self) -> str:
        return self.config.get("pipeline", {}).get("captioner", {}).get("backend", "qwen")

    def _should_caption(self, rec: Dict[str, Any], caption_all: bool) -> bool:
        if caption_all:
            return True
        return bool(rec.get("keep", True)) and not rec.get("reject")

    def _ensure_caption_frames(
        self,
        rec: Dict[str, Any],
        frames_dir: Path,
        actor_frames_dir: Path,
        multi_frame: bool,
    ) -> list[Path]:
        if multi_frame:
            frame_paths = pick_caption_frames(rec, frames_dir)
            if not frame_paths:
                frame_paths = pick_caption_frames(rec, actor_frames_dir)
        else:
            single = pick_caption_frame(rec, frames_dir) or pick_caption_frame(
                rec, actor_frames_dir
            )
            frame_paths = [single] if single else []
        if frame_paths or not self.movie_video:
            return frame_paths
        frames_dir.mkdir(parents=True, exist_ok=True)
        extract_clip_frames(self.movie_video, rec, frames_dir)
        if multi_frame:
            return pick_caption_frames(rec, frames_dir)
        single = pick_caption_frame(rec, frames_dir)
        return [single] if single else []

    def _process_gemma(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        pcfg = self.config.get("pipeline", {}).get("captioner", {})
        frames_dir = self.movie_dir / "frames"
        actor_frames_dir = self.movie_dir / "actor_frames"
        max_caption = self.config.get("_test", {}).get("max_clips")
        prompt_version = pcfg.get("prompt_version", "eros_structured_v1")
        multi_frame = bool(pcfg.get("multi_frame", True))
        caption_all = bool(pcfg.get("caption_all_clips", True))

        captioner = GemmaCaptionService.acquire(self.config)
        to_caption: List[tuple[Dict[str, Any], list[Path]]] = []
        skipped = 0

        for rec in records:
            if self.should_skip_clip(rec):
                continue
            if not self._should_caption(rec, caption_all):
                rec["caption"] = ""
                rec["generated_caption"] = ""
                rec["caption_struct"] = {}
                rec["prompt_version"] = ""
                MetadataManager.mark_done(rec, self.service_id)
                continue
            if max_caption and len(to_caption) >= max_caption:
                MetadataManager.mark_done(rec, self.service_id)
                continue

            frame_paths = self._ensure_caption_frames(
                rec, frames_dir, actor_frames_dir, multi_frame
            )
            if not frame_paths:
                rec["caption"] = ""
                rec["generated_caption"] = ""
                rec["caption_struct"] = {}
                rec["prompt_version"] = prompt_version
                skipped += 1
                MetadataManager.mark_done(rec, self.service_id)
                continue
            to_caption.append((rec, frame_paths))

        captioned = with_actors = 0
        try:
            batch_size = int(
                self.config.get("models", {}).get("gemma_caption", {}).get("batch_size", 8)
            )
            for start in range(0, len(to_caption), batch_size):
                batch = to_caption[start : start + batch_size]
                raw_caps = captioner.caption_records(batch)
                for (rec, _), raw in zip(batch, raw_caps):
                    gen_line = to_single_line_json(raw)
                    struct = parse_caption_json(raw)
                    rec["generated_caption"] = gen_line
                    rec["caption_struct"] = struct
                    rec["caption"] = struct.get("short_description") or gen_line
                    rec["prompt_version"] = prompt_version
                    if rec.get("clip_actors"):
                        with_actors += 1
                    captioned += 1
                    MetadataManager.mark_done(rec, self.service_id)
        finally:
            GemmaCaptionService.release()

        cc = self.config.get("models", {}).get("gemma_caption", {})
        return {
            "captioned": captioned,
            "skipped_no_frame": skipped,
            "captions_with_actor_names": with_actors,
            "model": "Gemma-3-4B-IT",
            "gpus": cc.get("gpu_ids", pcfg.get("gpu_ids", [0])),
            "backend": "gemma",
            "prompt_version": prompt_version,
            "caption_all_clips": caption_all,
            "frames_per_clip": 3 if multi_frame else 1,
        }

    def _process_qwen(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt_mgr = self.config.get("_prompt_manager")
        mp = self.config["pipeline"]["master_pipeline"]
        gpu_ids = [int(g) for g in mp.get("caption_gpu_ids", [0, 1, 2, 3])]

        init_master(mp["root"])
        master_prompts = load_master_prompts(
            Path(mp["root"]) / mp.get("prompts_dir", "prompts")
        )
        vlm = QwenVLMService.acquire(self.config, gpu_ids, "s8")
        frames_dir = self.movie_dir / "actor_frames"
        pcfg = self.config.get("pipeline", {}).get("captioner", {})
        caption_all = bool(pcfg.get("caption_all_clips", True))
        max_caption = self.config.get("_test", {}).get("max_clips")
        captioned = with_actors = captioned_limit = 0

        try:
            for rec in records:
                if self.should_skip_clip(rec):
                    continue
                if not self._should_caption(rec, caption_all):
                    rec["caption"] = ""
                    rec["generated_caption"] = ""
                    rec["caption_struct"] = {}
                    rec["prompt_version"] = ""
                    MetadataManager.mark_done(rec, self.service_id)
                    continue
                if max_caption and captioned_limit >= max_caption:
                    MetadataManager.mark_done(rec, self.service_id)
                    continue

                bucket = rec.get("bucket", "bucket_01")
                category = bucket_to_category(
                    bucket,
                    prompt_mgr.get_bucket_info(bucket)["slug"] if prompt_mgr else "",
                )
                if master_prompts:
                    bucket_prompt = master_prompts.get(
                        category, master_prompts.get("people_portraits", "")
                    )
                    prompt_version = f"master_{category}"
                elif prompt_mgr:
                    try:
                        bucket_prompt = prompt_mgr.get_prompt(bucket)
                        prompt_version = prompt_mgr.version
                    except KeyError:
                        bucket_prompt = prompt_mgr.get_prompt("bucket_01")
                        prompt_version = prompt_mgr.version
                else:
                    bucket_prompt = ""
                    prompt_version = ""

                actors = rec.get("actors") or []
                full_prompt = build_caption_prompt(bucket_prompt, actors)
                if actors:
                    with_actors += 1

                images = []
                if self.movie_video:
                    images = clip_keyframe_images(
                        self.movie_video, rec, self.config, [0.25, 0.5, 0.75]
                    )
                if not images:
                    frame_path = frames_dir / f"{rec['clip_id']}.jpg"
                    if not frame_path.exists() and self.movie_video:
                        save_clip_keyframe(self.movie_video, rec, frames_dir)
                    if frame_path.exists():
                        from PIL import Image
                        images = [Image.open(frame_path).convert("RGB")]

                if images:
                    raw = vlm.generate_multi(images, full_prompt)
                    data = parse_vlm_json(raw, {"caption": "", "tags": {}})
                else:
                    data = {"caption": "", "tags": {}}

                caption = caption_to_str(data.get("caption", ""))
                if actors:
                    caption = enforce_actor_names_in_caption(caption, actors)
                rec["caption"] = caption
                rec["generated_caption"] = caption
                rec["caption_struct"] = {k: v for k, v in data.items() if k != "caption"}
                rec["prompt_version"] = prompt_version
                captioned += 1
                captioned_limit += 1
                MetadataManager.mark_done(rec, self.service_id)

            self.metadata.write_all(records)
        finally:
            QwenVLMService.release()

        return {
            "captioned": captioned,
            "captions_with_actor_names": with_actors,
            "model": mp.get("vlm_model_name", "Qwen3-VL-32B"),
            "gpus": gpu_ids,
            "backend": "qwen",
            "caption_all_clips": caption_all,
            "frames_per_clip": 3,
        }

    def _process_qwen_video(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        pcfg = self.config.get("pipeline", {}).get("captioner", {})
        max_caption = self.config.get("_test", {}).get("max_clips")
        prompt_version = pcfg.get("prompt_version", "qwen_video_eros_v1")
        caption_all = bool(pcfg.get("caption_all_clips", True))
        clips_dir = self.movie_dir / "clips"

        captioner = QwenVideoCaptionService.acquire(self.config)
        to_caption: List[tuple[Dict[str, Any], Path]] = []
        skipped = 0

        if not self.movie_video:
            raise FileNotFoundError(f"No movie video in {self.movie_dir}")

        for rec in records:
            if self.should_skip_clip(rec):
                continue
            if not self._should_caption(rec, caption_all):
                rec["caption"] = ""
                rec["generated_caption"] = ""
                rec["caption_struct"] = {}
                rec["prompt_version"] = ""
                MetadataManager.mark_done(rec, self.service_id)
                continue
            if max_caption and len(to_caption) >= max_caption:
                MetadataManager.mark_done(rec, self.service_id)
                continue

            clip_path = ensure_clip_mp4(
                self.movie_video, rec, clips_dir, self.config
            )
            if not clip_path:
                rec["caption"] = ""
                rec["generated_caption"] = ""
                rec["caption_struct"] = {}
                rec["prompt_version"] = prompt_version
                skipped += 1
                MetadataManager.mark_done(rec, self.service_id)
                continue
            to_caption.append((rec, clip_path))

        captioned = with_actors = 0
        try:
            batch_size = int(
                self.config.get("models", {})
                .get("qwen_video_caption", {})
                .get("batch_size", pcfg.get("batch_size", 1))
            )
            for start in range(0, len(to_caption), batch_size):
                batch = to_caption[start : start + batch_size]
                raw_caps = captioner.caption_records(batch)
                for (rec, _), raw in zip(batch, raw_caps):
                    gen_line = to_single_line_json(raw)
                    struct = parse_caption_json(raw)
                    rec["generated_caption"] = gen_line
                    rec["caption_struct"] = struct
                    rec["caption"] = struct.get("short_description") or gen_line
                    rec["prompt_version"] = prompt_version
                    if rec.get("clip_actors"):
                        with_actors += 1
                    captioned += 1
                    MetadataManager.mark_done(rec, self.service_id)
        finally:
            QwenVideoCaptionService.release()

        qc = self.config.get("models", {}).get("qwen_video_caption", {})
        model_path = str(captioner.model_path)
        return {
            "captioned": captioned,
            "skipped_no_clip": skipped,
            "captions_with_actor_names": with_actors,
            "model": Path(model_path).name,
            "gpus": qc.get("gpu_ids", pcfg.get("gpu_ids", [0])),
            "backend": "qwen_video",
            "prompt_version": prompt_version,
            "caption_all_clips": caption_all,
            "video_fps": captioner.fps,
            "input_mode": "native_mp4",
        }

    def process_movie(self) -> Dict[str, Any]:
        records = self.metadata.read_all()
        backend = self._backend()
        if backend == "gemma":
            stats = self._process_gemma(records)
            self.metadata.write_all(records)
            return stats
        if backend == "qwen_video":
            stats = self._process_qwen_video(records)
            self.metadata.write_all(records)
            return stats
        return self._process_qwen(records)
