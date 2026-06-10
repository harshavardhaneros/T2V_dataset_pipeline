"""Service 1: scene detection, virtual clips, crop, phash."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import imagehash
from PIL import Image
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

from common.base_service import BaseService
from common.progress import iter_progress
from common.ffmpeg_utils import detect_crop
from common.metadata_manager import MetadataManager, new_clip_record


class ExtractService(BaseService):
    service_id = "s1"
    service_name = "s1_extract"
    owned_fields = [
        "video_id", "clip_id", "scene_id", "source_video",
        "timestamp_start", "timestamp_end", "duration", "phash", "crop_box",
    ]

    def _collect_clip_specs(
        self,
        scene_list,
        clip_length: float,
        time_offset: float,
        max_clips: int | None,
    ) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        clip_counter = 0
        for scene_id, (scene_start, scene_end) in enumerate(scene_list):
            start_sec = scene_start.get_seconds()
            end_sec = scene_end.get_seconds()
            current_start = start_sec
            while current_start + clip_length <= end_sec:
                clip_start = current_start
                clip_end = current_start + clip_length
                specs.append({
                    "clip_index": clip_counter,
                    "scene_id": scene_id,
                    "clip_start": clip_start,
                    "clip_end": clip_end,
                    "middle_time": (clip_start + clip_end) / 2.0,
                    "time_offset": time_offset,
                })
                clip_counter += 1
                if max_clips and clip_counter >= max_clips:
                    return specs
                current_start += clip_length
            if max_clips and clip_counter >= max_clips:
                break
        return specs

    def _compute_phashes_single_pass(
        self,
        video_path: Path,
        specs: List[Dict[str, Any]],
        fps: float,
        crop_parts: Tuple[int, int, int, int] | None,
    ) -> Dict[int, str]:
        """One forward video pass for all phash values (avoids per-clip seek)."""
        if not specs:
            return {}

        ordered = sorted(specs, key=lambda s: s["middle_time"])
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {s["clip_index"]: "" for s in specs}

        phashes: Dict[int, str] = {}
        frame_idx = 0
        for spec in iter_progress(ordered, desc="s1 phash", unit="clip"):
            target = max(0, int(spec["middle_time"] * fps))
            while frame_idx < target:
                if not cap.grab():
                    break
                frame_idx += 1
            ok, frame = cap.read()
            if not ok:
                phashes[spec["clip_index"]] = ""
                continue
            frame_idx += 1
            if crop_parts:
                cw, ch, cx, cy = crop_parts
                frame = frame[cy : cy + ch, cx : cx + cw]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            phashes[spec["clip_index"]] = str(imagehash.phash(Image.fromarray(frame_rgb)))

        cap.release()
        return phashes

    def process_movie(self) -> Dict[str, Any]:
        if not self.movie_video or not self.movie_video.exists():
            raise FileNotFoundError(f"No movie video in {self.movie_dir}")

        thresholds = self.config.get("thresholds", {})
        sd = thresholds.get("scene_detection", {})
        vc = thresholds.get("virtual_clips", {})
        cd = thresholds.get("crop_detect", {})

        threshold = float(sd.get("threshold", 27))
        min_scene_len = int(sd.get("min_scene_len", 15))
        clip_length = float(vc.get("clip_length_sec", 5))

        crop_box = detect_crop(
            str(self.movie_video),
            sample_seconds=int(cd.get("sample_seconds", 30)),
            random_seed=int(cd.get("random_seed", 42)),
        ) or ""

        video = open_video(str(self.movie_video))
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
        )
        scene_manager.detect_scenes(video=video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        video_id = self.movie_video.stem
        source_name = self.movie_video.name

        test_cfg = self.config.get("_test", {})
        max_clips = test_cfg.get("max_clips")
        time_offset = float(test_cfg.get("time_offset_sec", 0))

        specs = self._collect_clip_specs(
            scene_list, clip_length, time_offset, max_clips
        )

        cap = cv2.VideoCapture(str(self.movie_video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.movie_video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        crop_parts = None
        if crop_box:
            crop_parts = tuple(int(x) for x in crop_box.split(":"))

        phashes = self._compute_phashes_single_pass(
            self.movie_video, specs, fps, crop_parts
        )

        records: List[Dict[str, Any]] = []
        for spec in iter_progress(specs, desc="s1 clips", unit="clip"):
            idx = spec["clip_index"]
            clip_start = spec["clip_start"]
            clip_end = spec["clip_end"]
            rec = new_clip_record(
                video_id=video_id,
                clip_id=f"{video_id}_{idx:06d}",
                scene_id=spec["scene_id"],
                source_video=source_name,
                timestamp_start=round(clip_start + time_offset, 3),
                timestamp_end=round(clip_end + time_offset, 3),
                duration=clip_length,
                phash=phashes.get(idx, ""),
                crop_box=crop_box,
            )
            if time_offset:
                rec["segment_time_offset_sec"] = time_offset
            MetadataManager.mark_done(rec, self.service_id)
            records.append(rec)

        self.metadata.write_all(records, use_lock=True)
        return {
            "scenes_detected": len(scene_list),
            "clips_generated": len(records),
            "clip_length_sec": clip_length,
            "crop_box": crop_box,
        }
