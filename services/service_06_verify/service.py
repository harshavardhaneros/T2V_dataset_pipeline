"""Service 6: Gemma VLM verification (2nd pass on keyframe + bucket)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from common.base_service import BaseService
from common.gemma_verify import GemmaVerifyService
from common.gpu_info import log_service_gpus
from common.metadata_manager import MetadataManager
from common.vlm_service import clip_keyframe_images


class VerifyService(BaseService):
    service_id = "s6"
    service_name = "s6_verify"
    owned_fields = ["verified", "confidence", "route"]

    def process_movie(self) -> Dict[str, Any]:
        records = self.metadata.read_all()
        mp = self.config["pipeline"]["master_pipeline"]
        gcfg = self.config.get("models", {}).get("gemma", {})
        gpu_ids = [int(g) for g in gcfg.get("gpu_ids", mp.get("verify_gpu_ids", [5, 6]))]

        log_service_gpus(
            "s6",
            "VLM verify — Gemma 2nd pass",
            gcfg.get("model_path", mp.get("gemma_model_path", "NOT_SET")),
            gpu_ids,
        )

        gemma = GemmaVerifyService(self.config)
        verified_count = 0

        try:
            for rec in records:
                if self.should_skip_clip(rec):
                    continue
                if not rec.get("keep", True) or rec.get("reject"):
                    rec["verified"] = False
                    rec["confidence"] = 0.0
                    rec["route"] = "other"
                    MetadataManager.mark_done(rec, self.service_id)
                    continue

                images = []
                if self.movie_video:
                    images = clip_keyframe_images(
                        self.movie_video, rec, self.config, [0.5]
                    )

                if images:
                    data = gemma.verify(images[0], rec.get("bucket", "bucket_01"))
                else:
                    data = {"verified": False, "confidence": 0.0, "route": "other"}

                rec["verified"] = bool(data.get("verified", False))
                rec["confidence"] = float(data.get("confidence", 0.0))
                route = data.get("route", "other")
                bucket = str(rec.get("bucket", ""))
                if bucket in ("bucket_01", "people_portraits") or "people" in bucket:
                    route = "people"
                rec["route"] = route
                if rec["verified"]:
                    verified_count += 1
                MetadataManager.mark_done(rec, self.service_id)

            self.metadata.write_all(records)
        finally:
            gemma.cleanup()

        return {
            "verified_clips": verified_count,
            "model": gcfg.get("model_name", "Gemma"),
            "gpus": gpu_ids,
        }
