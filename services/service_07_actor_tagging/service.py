"""Service 7: 3-frame actor tagging + screen positions (eros-style)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from common.base_service import BaseService
from common.clip_io import clip_frame_path, extract_clip_frames
from common.gemma_caption import enrich_record_actor_fields
from common.gpu_info import log_service_gpus
from common.master_bridge import init_master, tag_actor_frames
from common.metadata_manager import MetadataManager
from common.paths import models_root, yolo_face_model_path
from common.screen_position import known_actor_names

logger = logging.getLogger(__name__)


class ActorTaggingService(BaseService):
    service_id = "s7"
    service_name = "s7_actor_tagging"
    owned_fields = [
        "actor_status", "actors", "clip_actors",
        "actors_f1", "actors_f2", "actors_f3",
        "pos_f1", "pos_f2", "pos_f3",
        "frame1", "frame2", "frame3",
    ]

    def _master_cfg(self) -> Dict[str, Any]:
        return self.config.get("pipeline", {}).get("master_pipeline", {})

    def process_movie(self) -> Dict[str, Any]:
        records = self.metadata.read_all()
        master_cfg = {**self._master_cfg(), "models_root": str(models_root(self.config))}
        if not master_cfg.get("root"):
            raise ValueError("pipeline.yaml: master_pipeline.root is required for s7")

        init_master(master_cfg["root"])
        log_service_gpus(
            "s7",
            "Face — YOLOv12n-face + InsightFace (3 frames/clip)",
            "Master_Pipeline_t2i_dataset/actor_tagger.py",
            [int(master_cfg.get("actor_tag_gpu_id", 4))],
            extra="embeddings: Master_Pipeline_t2i_dataset/actors/actor_embeddings",
        )

        frames_dir = self.movie_dir / "frames"
        actor_frames_dir = self.movie_dir / "actor_frames"
        tags_dir = self.movie_dir / "actor_tags"
        for d in (frames_dir, actor_frames_dir, tags_dir):
            d.mkdir(parents=True, exist_ok=True)

        to_tag: List[Dict[str, Any]] = []
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
            return {"people_clips": 0, "tagged": 0, "no_match": 0}

        if not self.movie_video:
            raise FileNotFoundError(f"No video in {self.movie_dir}")

        from common.master_bridge import ensure_yolo_face_model

        ensure_yolo_face_model(yolo_face_model_path(self.config))

        all_image_paths: List[Path] = []
        clip_frame_map: Dict[str, Dict[int, Path]] = {}

        for rec in to_tag:
            paths = extract_clip_frames(self.movie_video, rec, frames_dir)
            frame_map = {}
            for i in (1, 2, 3):
                p = clip_frame_path(frames_dir, rec["clip_id"], i)
                if p.exists():
                    frame_map[i] = p
                    all_image_paths.append(p)
            clip_frame_map[rec["clip_id"]] = frame_map

        tag_results = tag_actor_frames(all_image_paths, master_cfg, tags_dir)

        tagged = no_match = 0
        for rec in to_tag:
            frame_map = clip_frame_map.get(rec["clip_id"], {})
            frame_assignments: Dict[int, List[Dict[str, Any]]] = {}
            for idx, fp in frame_map.items():
                frame_assignments[idx] = tag_results.get(str(fp), [])

            enrich_record_actor_fields(rec, frame_assignments, frame_map)
            if rec.get("clip_actors"):
                rec["actor_status"] = "tagged"
                tagged += 1
            else:
                rec["actor_status"] = "no_match"
                no_match += 1
            MetadataManager.mark_done(rec, self.service_id)

        self.metadata.write_all(records)
        return {
            "people_clips": len(to_tag),
            "frames_extracted": len(all_image_paths),
            "tagged": tagged,
            "no_match": no_match,
            "frames_per_clip": 3,
            "actor_tag_gpu": master_cfg.get("actor_tag_gpu_id", 0),
        }
