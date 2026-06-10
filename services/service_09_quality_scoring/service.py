"""Service 9: pragmatic quality scoring — DOVER, motion, CLIP, bucket verify, caption."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from common.base_service import BaseService
from common.progress import iter_progress, ray_get_progress
from common.caption_text import caption_to_str
from common.frame_sampler import sample_keyframes
from common.gpu_actor_pool import ray_worker_count
from common.metadata_manager import MetadataManager
from common.ray_pool import init_ray, ray_settings
from common.video_time import clip_local_range
from model_clients.clip_client import ClipClient


class QualityScoringService(BaseService):
    service_id = "s9"
    service_name = "s9_quality_scoring"
    owned_fields = ["clip_score", "icr", "aod", "final_score"]

    def _score_cfg(self) -> Dict[str, Any]:
        return self.config.get("thresholds", {}).get("quality_scoring", {})

    def _frame_fractions(self) -> List[float]:
        fracs = self._score_cfg().get("score_frame_fractions")
        if fracs:
            return [float(f) for f in fracs]
        return [0.1, 0.3, 0.5, 0.7, 0.9]

    def _clip_length_sec(self) -> float:
        return float(
            self.config.get("thresholds", {}).get("virtual_clips", {}).get("clip_length_sec", 5)
        )

    def _use_ray(self) -> bool:
        return bool(ray_settings(self.config).get("parallel_clip_score", False))

    def _weights(self) -> Dict[str, float]:
        weights = self.config.get("thresholds", {}).get("quality_weights", {})
        return {
            "clip_score": float(weights.get("clip_score", 0.25)),
            "dover_score": float(weights.get("dover_score", 0.30)),
            "motion_score": float(weights.get("motion_score", 0.20)),
            "bucket_semantic": float(weights.get("bucket_semantic", 0.15)),
            "caption_present": float(weights.get("caption_present", 0.10)),
        }

    def _bucket_semantic(self, rec: Dict[str, Any]) -> float:
        verified = rec.get("bucket_verified")
        if verified is None:
            verified = rec.get("verified", False)
        if not verified:
            return 0.0
        return max(0.0, min(1.0, float(rec.get("bucket_confidence", 0) or 0)))

    def _dover_score(self, rec: Dict[str, Any]) -> float:
        if rec.get("dover_score") is not None:
            return max(0.0, min(1.0, float(rec["dover_score"])))
        if rec.get("aesthetic_score") is not None:
            return max(0.0, min(1.0, float(rec["aesthetic_score"])))
        return 0.0

    def _motion_score(self, rec: Dict[str, Any]) -> float:
        if rec.get("motion_score") is None:
            return 0.0
        return max(0.0, min(1.0, float(rec["motion_score"])))

    def _caption_present(self, rec: Dict[str, Any]) -> float:
        cap = caption_to_str(rec.get("caption"))
        return 1.0 if cap and cap.strip() else 0.0

    def _score_one(
        self,
        rec: Dict[str, Any],
        clip_client: ClipClient,
        clip_path: Path | None,
    ) -> Dict[str, float]:
        weights = self._weights()
        fractions = self._frame_fractions()
        clip_len = self._clip_length_sec()
        caption = caption_to_str(rec.get("caption")) or "Indic cultural scene"

        if clip_path and clip_path.exists():
            frames = sample_keyframes(str(clip_path), 0.0, clip_len, fractions=fractions)
        elif self.movie_video:
            start, end = clip_local_range(rec, self.config)
            frames = sample_keyframes(
                str(self.movie_video),
                start,
                end,
                fractions=fractions,
                crop_box=rec.get("crop_box", ""),
            )
        else:
            frames = []

        clip_score = 0.0
        if frames:
            scores = [clip_client.score_image_text(frame, caption) for frame in frames]
            clip_score = sum(scores) / len(scores)

        bucket_sem = self._bucket_semantic(rec)
        dover = self._dover_score(rec)
        motion = self._motion_score(rec)
        cap_ok = self._caption_present(rec)
        final = (
            weights["clip_score"] * clip_score
            + weights["dover_score"] * dover
            + weights["motion_score"] * motion
            + weights["bucket_semantic"] * bucket_sem
            + weights["caption_present"] * cap_ok
        )
        return {
            "clip_score": round(clip_score, 4),
            "icr": round(bucket_sem, 4),
            "aod": round(dover, 4),
            "final_score": round(final, 4),
        }

    def _score_ray(
        self,
        targets: List[Dict[str, Any]],
        clips_dir: Path,
    ) -> Dict[str, Dict[str, float]]:
        from common.vlm_ray_actors import ClipScoreActor

        if ClipScoreActor is None or not init_ray(self.config):
            return {}

        import ray

        clip_cfg = self.config.get("models", {}).get("clip_model", {})
        gpu_ids = [int(g) for g in clip_cfg.get("gpu_ids", [0])]
        n_actors = ray_worker_count(self.config, "clip_score_workers", gpu_ids)
        weights = self._weights()
        fractions = self._frame_fractions()
        clip_len = self._clip_length_sec()

        actors = [ClipScoreActor.remote(self.config) for _ in range(n_actors)]
        payloads = [
            {
                "record": rec,
                "clip_path": str(clips_dir / f"{rec['clip_id']}.mp4"),
                "weights": weights,
                "fractions": fractions,
                "clip_length_sec": clip_len,
            }
            for rec in targets
        ]
        futures = [
            actors[i % n_actors].score_clip.remote(payload)
            for i, payload in enumerate(payloads)
        ]
        try:
            rows = ray_get_progress(futures, desc="s9 score")
        finally:
            for actor in actors:
                try:
                    ray.kill(actor)
                except Exception:
                    pass
        return {row["clip_id"]: row for row in rows}

    def process_movie(self) -> Dict[str, Any]:
        records = self.metadata.read_all()
        weights = self._weights()
        fractions = self._frame_fractions()
        clips_dir = self.movie_dir / "clips"

        targets = [
            rec for rec in records if not self.should_skip_clip(rec) and rec.get("keep", True)
        ]

        scores_by_id: Dict[str, Dict[str, float]] = {}
        if self._use_ray() and len(targets) >= int(
            ray_settings(self.config).get("parallel_clip_min", 2)
        ):
            scores_by_id = self._score_ray(targets, clips_dir)

        clip_client: ClipClient | None = None
        if not scores_by_id:
            clip_client = ClipClient(self.config.get("models", {}))
            for rec in iter_progress(targets, desc="s9 score", unit="clip"):
                clip_path = clips_dir / f"{rec['clip_id']}.mp4"
                scores_by_id[rec["clip_id"]] = self._score_one(rec, clip_client, clip_path)

        scored = 0
        for rec in iter_progress(records, desc="s9 apply", unit="clip"):
            if self.should_skip_clip(rec):
                continue
            if not rec.get("keep", True):
                rec["clip_score"] = 0.0
                rec["icr"] = 0.0
                rec["aod"] = 0.0
                rec["final_score"] = 0.0
                MetadataManager.mark_done(rec, self.service_id)
                continue

            row = scores_by_id.get(rec["clip_id"], {})
            rec["clip_score"] = row.get("clip_score", 0.0)
            rec["icr"] = row.get("icr", 0.0)
            rec["aod"] = row.get("aod", 0.0)
            rec["final_score"] = row.get("final_score", 0.0)
            scored += 1
            MetadataManager.mark_done(rec, self.service_id)

        placeholder = False
        if clip_client is not None:
            placeholder = clip_client.use_placeholder

        self.metadata.write_all(records)
        return {
            "scored": scored,
            "clip_frames": len(fractions),
            "clip_model_placeholder": placeholder,
            "weights": weights,
            "ray": bool(self._use_ray() and scores_by_id),
        }
