"""Service 6: route verification — fast bucket_route or optional Gemma VLM."""

from __future__ import annotations

from typing import Any, Dict, List

from common.base_service import BaseService
from common.gemma_verify import GemmaVerifyService
from common.gpu_info import log_service_gpus
from common.metadata_manager import MetadataManager
from common.vlm_service import clip_keyframe_images


def _route_from_bucket(bucket: str) -> str:
    b = str(bucket or "").lower()
    if b in ("bucket_01", "people_portraits") or "people" in b:
        return "people"
    return "other"


class VerifyService(BaseService):
    service_id = "s6"
    service_name = "s6_verify"
    owned_fields = ["verified", "confidence", "route", "bucket_verified"]

    def _s6_cfg(self) -> Dict[str, Any]:
        return self.config.get("pipeline", {}).get("s6", {})

    def _apply_bucket_route(self, rec: Dict[str, Any]) -> None:
        bucket = str(rec.get("bucket", ""))
        conf = float(rec.get("bucket_confidence", 0.5) or 0.5)
        rec["verified"] = not rec.get("reject", False)
        rec["confidence"] = conf
        rec["route"] = _route_from_bucket(bucket)

    def process_movie(self) -> Dict[str, Any]:
        records = self.metadata.read_all()
        s6 = self._s6_cfg()
        mode = s6.get("mode", "bucket_route")
        mp = self.config["pipeline"]["master_pipeline"]
        gcfg = self.config.get("models", {}).get("gemma", {})
        gpu_ids = [int(g) for g in gcfg.get("gpu_ids", mp.get("verify_gpu_ids", [0]))]

        if mode == "bucket_route":
            print(
                "[s6] bucket_route mode — no VLM load (route derived from s5 bucket)",
                flush=True,
            )
            verified_count = 0
            for rec in records:
                if self.should_skip_clip(rec):
                    continue
                if not rec.get("keep", True) or rec.get("reject"):
                    rec["verified"] = False
                    rec["confidence"] = 0.0
                    rec["route"] = "other"
                else:
                    self._apply_bucket_route(rec)
                    if rec["verified"]:
                        verified_count += 1
                MetadataManager.mark_done(rec, self.service_id)
            self.metadata.write_all(records)
            return {
                "verified_clips": verified_count,
                "model": "bucket_route",
                "gpus": [],
                "mode": mode,
            }

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
                    rec["bucket_verified"] = False
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
                    data = {
                        "verified": False,
                        "confidence": 0.0,
                        "route": "other",
                        "bucket_matches": False,
                    }

                bucket_ok = bool(data.get("bucket_matches", data.get("verified", False)))
                rec["bucket_verified"] = bucket_ok
                rec["verified"] = bucket_ok
                rec["confidence"] = float(data.get("confidence", 0.0))
                route = str(data.get("route", "other"))
                rec["route"] = "people" if route == "people" else _route_from_bucket(
                    rec.get("bucket", "")
                )
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
            "mode": mode,
        }
