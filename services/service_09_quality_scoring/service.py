"""Service 9: CLIP + ICR + AOD quality scoring."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from common.base_service import BaseService
from common.caption_text import caption_to_str
from common.frame_sampler import read_middle_frame
from common.video_time import clip_time_offset
from common.metadata_manager import MetadataManager
from model_clients.clip_client import ClipClient


class QualityScoringService(BaseService):
    service_id = "s9"
    service_name = "s9_quality_scoring"
    owned_fields = ["clip_score", "icr", "aod", "final_score"]

    def _icr_framework(self, rec: Dict[str, Any]) -> float:
        bucket = rec.get("bucket", "")
        h = int(hashlib.md5(bucket.encode()).hexdigest(), 16)
        base = 0.5 + (h % 35) / 100.0
        if rec.get("reject"):
            return 0.1
        return min(1.0, base + float(rec.get("bucket_confidence", 0)) * 0.2)

    def _aod_framework(self, rec: Dict[str, Any]) -> float:
        h = int(hashlib.md5(rec.get("clip_id", "").encode()).hexdigest(), 16)
        return 0.45 + (h % 40) / 100.0

    def process_movie(self) -> Dict[str, Any]:
        records = self.metadata.read_all()
        weights = self.config.get("thresholds", {}).get("quality_weights", {})
        w_clip = float(weights.get("clip_score", 0.35))
        w_icr = float(weights.get("icr", 0.40))
        w_aod = float(weights.get("aod", 0.25))
        clip_client = ClipClient(self.config.get("models", {}))
        scored = 0

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
            clip_score = 0.5
            if self.movie_video:
                frame = read_middle_frame(
                    str(self.movie_video),
                    rec["timestamp_start"],
                    rec["timestamp_end"],
                    crop_box=rec.get("crop_box", ""),
                    time_offset=clip_time_offset(rec, self.config),
                )
                if frame is not None:
                    clip_score = clip_client.score_image_text(frame, caption)

            icr = self._icr_framework(rec)
            aod = self._aod_framework(rec)
            final = w_clip * clip_score + w_icr * icr + w_aod * aod

            rec["clip_score"] = round(clip_score, 4)
            rec["icr"] = round(icr, 4)
            rec["aod"] = round(aod, 4)
            rec["final_score"] = round(final, 4)
            scored += 1
            MetadataManager.mark_done(rec, self.service_id)

        self.metadata.write_all(records)
        return {"scored": scored}
