"""Service 7: 3-frame actor tagging + screen positions (eros-style)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from common.base_service import BaseService
from common.progress import iter_progress, ray_get_progress
from common.clip_io import clip_frame_path, extract_clip_frames
from common.clip_workers import extract_clip_frames_job
from common.actor_caption import finalize_actor_status
from common.gemma_caption import enrich_record_actor_fields
from common.gpu_actor_pool import ray_worker_count
from common.gpu_info import log_service_gpus
from common.master_bridge import init_master, tag_actor_frames
from common.metadata_manager import MetadataManager
from common.paths import master_pipeline_root, models_root, yolo_face_model_path
from common.ray_pool import init_ray, parallel_map, ray_settings, shutdown_ray

logger = logging.getLogger(__name__)


class ActorTaggingService(BaseService):
    service_id = "s7"
    service_name = "s7_actor_tagging"
    owned_fields = [
        "actor_status", "actors", "clip_actors",
        "actor_tag_min_similarity", "actor_tag_min_margin",
        "actors_f1", "actors_f2", "actors_f3",
        "pos_f1", "pos_f2", "pos_f3",
        "frame1", "frame2", "frame3",
    ]

    def _master_cfg(self) -> Dict[str, Any]:
        return self.config.get("pipeline", {}).get("master_pipeline", {})

    def _tag_frame_indices(self) -> List[int]:
        """Frame indices used for actor tagging (default 1,2,3; use [2] for ~3× faster)."""
        mp = self._master_cfg()
        indices = mp.get("actor_tag_frame_indices")
        if indices is None:
            return [1, 2, 3]
        return [int(i) for i in indices]

    def _use_ray_tag(self) -> bool:
        return bool(ray_settings(self.config).get("parallel_actor_tag", False))

    def _tag_ray(
        self,
        to_tag: List[Dict[str, Any]],
        clip_frame_map: Dict[str, Dict[int, Path]],
        master_cfg: Dict[str, Any],
        tags_dir: Path,
    ) -> Dict[str, Dict[str, Any]]:
        from common.vlm_ray_actors import ActorTagActor

        if ActorTagActor is None or not init_ray(self.config):
            return {}

        import ray

        mp = master_cfg
        gpu_ids = [int(mp.get("actor_tag_gpu_id", 0))]
        n_actors = ray_worker_count(self.config, "actor_tag_workers", gpu_ids)
        serializable_master = {**master_cfg, "models_root": str(master_cfg.get("models_root", ""))}
        actors = [
            ActorTagActor.remote(self.config, serializable_master) for _ in range(n_actors)
        ]
        payloads = []
        for rec in to_tag:
            frame_map = clip_frame_map.get(rec["clip_id"], {})
            payloads.append({
                "record": rec,
                "tags_dir": str(tags_dir),
                "frame_map": {str(k): str(v) for k, v in frame_map.items()},
            })
        chunks: List[List[Dict[str, Any]]] = [[] for _ in range(n_actors)]
        for i, payload in enumerate(payloads):
            chunks[i % n_actors].append(payload)
        futures = [
            actors[i].tag_clips_batch.remote(chunk)
            for i, chunk in enumerate(chunks)
            if chunk
        ]
        try:
            rows: List[Dict[str, Any]] = []
            for batch_rows in ray_get_progress(futures, desc="s7 actor tag"):
                rows.extend(batch_rows)
        finally:
            for actor in actors:
                try:
                    ray.kill(actor)
                except Exception:
                    pass
        return {row["clip_id"]: row for row in rows}

    def process_movie(self) -> Dict[str, Any]:
        records = self.metadata.read_all()
        master_cfg = {**self._master_cfg(), "models_root": str(models_root(self.config))}
        if not master_cfg.get("root"):
            raise ValueError("pipeline.yaml: master_pipeline.root is required for s7")

        init_master(master_pipeline_root(self.config))
        log_service_gpus(
            "s7",
            "Face — YOLOv12n-face + InsightFace (3 frames/clip)",
            str(master_pipeline_root(self.config) / "actor_tagger.py"),
            [int(master_cfg.get("actor_tag_gpu_id", 0))],
            extra="Ray multi-GPU" if self._use_ray_tag() else "",
        )

        frames_dir = self.movie_dir / "frames"
        actor_frames_dir = self.movie_dir / "actor_frames"
        tags_dir = self.movie_dir / "actor_tags"
        for d in (frames_dir, actor_frames_dir, tags_dir):
            d.mkdir(parents=True, exist_ok=True)

        caption_all = bool(
            self.config.get("pipeline", {}).get("captioner", {}).get("caption_all_clips", True)
        )
        to_tag: List[Dict[str, Any]] = []
        frames_extracted = 0

        if not self.movie_video:
            raise FileNotFoundError(f"No video in {self.movie_dir}")

        extract_targets: List[Dict[str, Any]] = []
        for rec in records:
            if self.should_skip_clip(rec):
                continue
            if caption_all:
                extract_targets.append(rec)

        parallel_frames = bool(ray_settings(self.config).get("parallel_frame_extract", True))
        frame_workers = int(ray_settings(self.config).get("frame_extract_workers", 0)) or None
        if extract_targets:
            if parallel_frames and len(extract_targets) >= int(
                ray_settings(self.config).get("parallel_clip_min", 4)
            ):
                jobs = [
                    {
                        "video_path": str(self.movie_video),
                        "record": rec,
                        "frames_dir": str(frames_dir),
                    }
                    for rec in extract_targets
                ]
                counts = parallel_map(
                    self.config,
                    extract_clip_frames_job,
                    jobs,
                    label="s7_frame_extract",
                    workers=frame_workers,
                )
                frames_extracted = sum(counts)
            else:
                for rec in iter_progress(
                    extract_targets, desc="s7 frame extract", unit="clip"
                ):
                    frames_extracted += len(
                        extract_clip_frames(self.movie_video, rec, frames_dir)
                    )

        for rec in records:
            if self.should_skip_clip(rec):
                continue

            if not rec.get("keep", True) or rec.get("reject"):
                rec["actor_status"] = "skipped"
                rec["actors"] = []
                rec["clip_actors"] = []
                MetadataManager.mark_done(rec, self.service_id)
                continue
            if rec.get("route") != "people":
                rec["actor_status"] = "not_applicable"
                rec["actors"] = []
                rec["clip_actors"] = []
                MetadataManager.mark_done(rec, self.service_id)
                continue
            to_tag.append(rec)

        if not to_tag:
            self.metadata.write_all(records)
            return {
                "people_clips": 0,
                "tagged": 0,
                "no_match": 0,
                "frames_extracted": frames_extracted,
                "frames_per_clip": 3,
                "caption_all_clips": caption_all,
            }

        from common.master_bridge import ensure_yolo_face_model

        ensure_yolo_face_model(yolo_face_model_path(self.config))

        tag_indices = self._tag_frame_indices()
        clip_frame_map: Dict[str, Dict[int, Path]] = {}
        all_image_paths: List[Path] = []

        for rec in iter_progress(to_tag, desc="s7 frame map", unit="clip"):
            if not caption_all:
                paths = extract_clip_frames(self.movie_video, rec, frames_dir)
                frames_extracted += len(paths)
            frame_map = {}
            for i in tag_indices:
                p = clip_frame_path(frames_dir, rec["clip_id"], i)
                if p.exists():
                    frame_map[i] = p
                    all_image_paths.append(p)
            clip_frame_map[rec["clip_id"]] = frame_map

        tag_rows: Dict[str, Dict[str, Any]] = {}
        if self._use_ray_tag() and len(to_tag) >= int(
            ray_settings(self.config).get("parallel_clip_min", 2)
        ):
            tag_rows = self._tag_ray(to_tag, clip_frame_map, master_cfg, tags_dir)

        if not tag_rows:
            tag_results = tag_actor_frames(all_image_paths, master_cfg, tags_dir)
            for rec in to_tag:
                frame_map = clip_frame_map.get(rec["clip_id"], {})
                frame_assignments = {
                    idx: tag_results.get(str(fp), []) for idx, fp in frame_map.items()
                }
                tag_rows[rec["clip_id"]] = {
                    "frame_assignments": frame_assignments,
                    "frame_map": {k: str(v) for k, v in frame_map.items()},
                }

        tagged = no_match = low_confidence = 0
        for rec in iter_progress(to_tag, desc="s7 apply tags", unit="clip"):
            row = tag_rows.get(rec["clip_id"], {})
            frame_assignments = row.get("frame_assignments", {})
            frame_map = {
                int(k): Path(v) for k, v in row.get("frame_map", {}).items()
            } or clip_frame_map.get(rec["clip_id"], {})

            enrich_record_actor_fields(rec, frame_assignments, frame_map)
            finalize_actor_status(rec, master_cfg)
            status = rec.get("actor_status")
            if status == "tagged":
                tagged += 1
            elif status == "low_confidence":
                low_confidence += 1
            else:
                no_match += 1
            MetadataManager.mark_done(rec, self.service_id)

        shutdown_ray(self.config)
        self.metadata.write_all(records)
        return {
            "people_clips": len(to_tag),
            "frames_extracted": frames_extracted or len(all_image_paths),
            "tagged": tagged,
            "low_confidence": low_confidence,
            "no_match": no_match,
            "frames_per_clip": len(tag_indices),
            "tag_frame_indices": tag_indices,
            "caption_all_clips": caption_all,
            "actor_tag_gpu": master_cfg.get("actor_tag_gpu_id", 0),
            "ray": bool(self._use_ray_tag() and tag_rows),
        }
