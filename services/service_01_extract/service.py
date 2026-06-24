"""Service 1: scene detection, virtual clips, crop, phash."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import imagehash
from PIL import Image
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode

from common.base_service import BaseService
from common.progress import iter_progress
from common.ffmpeg_utils import detect_crop
from common.metadata_manager import MetadataManager, new_clip_record


def _detect_scenes_in_range(
    video_path: str,
    fps: float,
    start_sec: float,
    end_sec: float,
    threshold: float,
    min_scene_len: int,
) -> List[Tuple[float, float]]:
    """Detect scenes within [start_sec, end_sec] (absolute movie time)."""
    if end_sec <= start_sec:
        return []
    start_tc = FrameTimecode(timecode=start_sec, fps=fps)
    end_tc = FrameTimecode(timecode=end_sec, fps=fps)
    # scenedetect 0.7 Cv2 backend does not accept start_time/end_time on open_video;
    # seek + detect_scenes(end_time=...) works and returns absolute timestamps.
    video = open_video(video_path)
    video.seek(start_tc)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    )
    scene_manager.detect_scenes(video=video, end_time=end_tc, show_progress=False)
    scene_list = scene_manager.get_scene_list()
    return [
        (scene_start.get_seconds(), scene_end.get_seconds())
        for scene_start, scene_end in scene_list
    ]


def _merge_scene_ranges(
    ranges: List[Tuple[float, float]], merge_gap_sec: float = 0.5
) -> List[Tuple[float, float]]:
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda r: r[0])
    merged: List[Tuple[float, float]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + merge_gap_sec:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


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

    def _video_duration_sec(self, video_path: Path, fps: float) -> float:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if frames > 0 and fps > 0:
            return float(frames) / fps
        return 0.0

    def _detect_scenes(
        self,
        video_path: Path,
        threshold: float,
        min_scene_len: int,
        fps: float,
    ) -> list:
        s1_cfg = self.config.get("pipeline", {}).get("s1", {})
        parallel_chunks = int(s1_cfg.get("parallel_scene_chunks", 0) or 0)
        overlap_sec = float(s1_cfg.get("scene_chunk_overlap_sec", 15))

        if parallel_chunks <= 1:
            video = open_video(str(video_path))
            scene_manager = SceneManager()
            scene_manager.add_detector(
                ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
            )
            scene_manager.detect_scenes(video=video, show_progress=False)
            return scene_manager.get_scene_list()

        duration = self._video_duration_sec(video_path, fps)
        if duration <= 0:
            parallel_chunks = 1
        if parallel_chunks <= 1:
            video = open_video(str(video_path))
            scene_manager = SceneManager()
            scene_manager.add_detector(
                ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
            )
            scene_manager.detect_scenes(video=video, show_progress=False)
            return scene_manager.get_scene_list()

        chunk_len = duration / parallel_chunks
        tasks: List[Tuple[str, float, float, float, float, int]] = []
        for i in range(parallel_chunks):
            start = max(0.0, i * chunk_len - (overlap_sec if i else 0.0))
            end = min(duration, (i + 1) * chunk_len + overlap_sec)
            tasks.append(
                (str(video_path), fps, start, end, threshold, min_scene_len)
            )

        ranges: List[Tuple[float, float]] = []
        with ProcessPoolExecutor(max_workers=parallel_chunks) as pool:
            futures = [pool.submit(_detect_scenes_in_range, *task) for task in tasks]
            for fut in as_completed(futures):
                ranges.extend(fut.result())

        merged = _merge_scene_ranges(ranges)
        scene_list = []
        for start_sec, end_sec in merged:
            scene_list.append(
                (
                    FrameTimecode(timecode=start_sec, fps=fps),
                    FrameTimecode(timecode=end_sec, fps=fps),
                )
            )
        return scene_list

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

        s1_cfg = self.config.get("pipeline", {}).get("s1", {})
        parallel_chunks = int(s1_cfg.get("parallel_scene_chunks", 0) or 0)

        cap = cv2.VideoCapture(str(self.movie_video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.movie_video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        with ThreadPoolExecutor(max_workers=2) as pool:
            crop_future = pool.submit(
                detect_crop,
                str(self.movie_video),
                int(cd.get("sample_seconds", 30)),
                int(cd.get("random_seed", 42)),
            )
            scenes_future = pool.submit(
                self._detect_scenes,
                self.movie_video,
                threshold,
                min_scene_len,
                fps,
            )
            crop_box = crop_future.result() or ""
            scene_list = scenes_future.result()

        video_id = self.movie_video.stem
        source_name = self.movie_video.name

        test_cfg = self.config.get("_test", {})
        max_clips = test_cfg.get("max_clips")
        time_offset = float(test_cfg.get("time_offset_sec", 0))

        specs = self._collect_clip_specs(
            scene_list, clip_length, time_offset, max_clips
        )

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
            "parallel_scene_chunks": parallel_chunks,
        }
