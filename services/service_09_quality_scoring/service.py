"""Service 9: pragmatic quality scoring — DOVER, motion, CLIP, bucket verify, caption."""

from __future__ import annotations

from typing import Any, Dict, List

from common.base_service import BaseService
from common.caption_text import caption_to_str
from common.frame_sampler import sample_keyframes
from common.metadata_manager import MetadataManager
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

    def _clip_alignment(
        self,
        rec: Dict[str, Any],
        caption: str,
        clip_client: ClipClient,
    ) -> float:
        if not self.movie_video:
            return 0.0
        start, end = clip_local_range(rec, self.config)
        frames = sample_keyframes(
            str(self.movie_video),
            start,
            end,
            fractions=self._frame_fractions(),
            crop_box=rec.get("crop_box", ""),
        )
        if not frames:
            return 0.0
        scores = [clip_client.score_image_text(frame, caption) for frame in frames]
        return sum(scores) / len(scores)

    def process_movie(self) -> Dict[str, Any]:
        records = self.metadata.read_all()
        weights = self.config.get("thresholds", {}).get("quality_weights", {})
        w_clip = float(weights.get("clip_score", 0.25))
        w_dover = float(weights.get("dover_score", 0.30))
        w_motion = float(weights.get("motion_score", 0.20))
        w_bucket = float(weights.get("bucket_semantic", 0.15))
        w_caption = float(weights.get("caption_present", 0.10))

        clip_client = ClipClient(self.config.get("models", {}))
        scored = 0
        fractions = self._frame_fractions()

        for rec in records:
            if self.should_skip_clip(rec):
                continue
            if not rec.get("keep", True):
                rec["clip_score"] = 0.0
                rec["icr"] = 0.0
                rec["aod"] = 0.0
                rec["final_score"] = 0.0
                MetadataManager.mark_done(rec, self.service_id)
                continue

            caption = caption_to_str(rec.get("caption")) or "Indic cultural scene"
            clip_score = self._clip_alignment(rec, caption, clip_client)
            bucket_sem = self._bucket_semantic(rec)
            dover = self._dover_score(rec)
            motion = self._motion_score(rec)
            cap_ok = self._caption_present(rec)

            final = (
                w_clip * clip_score
                + w_dover * dover
                + w_motion * motion
                + w_bucket * bucket_sem
                + w_caption * cap_ok
            )

            rec["clip_score"] = round(clip_score, 4)
            rec["icr"] = round(bucket_sem, 4)
            rec["aod"] = round(dover, 4)
            rec["final_score"] = round(final, 4)
            scored += 1
            MetadataManager.mark_done(rec, self.service_id)

        self.metadata.write_all(records)
        return {
            "scored": scored,
            "clip_frames": len(fractions),
            "clip_model_placeholder": clip_client.use_placeholder,
            "weights": {
                "clip_score": w_clip,
                "dover_score": w_dover,
                "motion_score": w_motion,
                "bucket_semantic": w_bucket,
                "caption_present": w_caption,
            },
        }
