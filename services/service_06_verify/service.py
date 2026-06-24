"""Service 6: route verification — fast bucket_route or optional Gemma VLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from common.base_service import BaseService
from common.progress import iter_progress, ray_get_progress
from common.gemma_verify import GemmaVerifyService
from common.gpu_actor_pool import ray_worker_count
from common.gpu_info import log_service_gpus
from common.metadata_manager import MetadataManager
from common.ray_pool import init_ray, ray_settings, shutdown_ray
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

    def _use_ray(self) -> bool:
        rc = ray_settings(self.config)
        return bool(rc.get("parallel_gpu_verify", False))

    def _apply_bucket_route(self, rec: Dict[str, Any]) -> None:
        bucket = str(rec.get("bucket", ""))
        conf = float(rec.get("bucket_confidence", 0.5) or 0.5)
        rec["verified"] = not rec.get("reject", False)
        rec["confidence"] = conf
        rec["route"] = _route_from_bucket(bucket)

    def _apply_verify_row(self, rec: Dict[str, Any], data: Dict[str, Any]) -> None:
        bucket_ok = bool(data.get("bucket_matches", data.get("verified", False)))
        rec["bucket_verified"] = bucket_ok
        rec["verified"] = bucket_ok
        rec["confidence"] = float(data.get("confidence", 0.0))
        route = str(data.get("route", "other"))
        rec["route"] = (
            "people"
            if route == "people"
            else _route_from_bucket(rec.get("bucket", ""))
        )

    def _verify_ray(self, targets: List[Dict[str, Any]], clips_dir: Path) -> Dict[str, Dict[str, Any]]:
        from common.vlm_ray_actors import GemmaVerifyActor

        if GemmaVerifyActor is None or not init_ray(self.config):
            return {}

        import ray

        from common.gpu_actor_pool import gpu_actor_options

        mp = self.config["pipeline"]["master_pipeline"]
        gpu_ids = [int(g) for g in mp.get("verify_gpu_ids", [0])]
        n_actors = ray_worker_count(self.config, "verify_workers", gpu_ids)
        opts = gpu_actor_options(self.config)
        actors = [GemmaVerifyActor.options(**opts).remote(self.config) for _ in range(n_actors)]
        payloads = [
            {
                "record": rec,
                "clip_path": str(clips_dir / f"{rec['clip_id']}.mp4"),
            }
            for rec in targets
        ]
        futures = [
            actors[i % n_actors].verify.remote(payload)
            for i, payload in enumerate(payloads)
        ]
        try:
            rows = ray_get_progress(futures, desc="s6 verify")
        finally:
            for actor in actors:
                try:
                    ray.kill(actor)
                except Exception:
                    pass
        return {row["clip_id"]: row for row in rows}

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
            for rec in iter_progress(records, desc="s6 route", unit="clip"):
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
            extra="Ray multi-GPU" if self._use_ray() else "",
        )

        clips_dir = self.movie_dir / "clips"
        targets = [
            rec
            for rec in records
            if not self.should_skip_clip(rec) and rec.get("keep", True) and not rec.get("reject")
        ]

        verified_count = 0
        results_by_id: Dict[str, Dict[str, Any]] = {}

        if self._use_ray() and len(targets) >= int(ray_settings(self.config).get("parallel_clip_min", 2)):
            results_by_id = self._verify_ray(targets, clips_dir)

        gemma: GemmaVerifyService | None = None
        try:
            if not results_by_id:
                gemma = GemmaVerifyService(self.config)
                for rec in iter_progress(targets, desc="s6 verify", unit="clip"):
                    images = []
                    clip_path = clips_dir / f"{rec['clip_id']}.mp4"
                    if clip_path.exists():
                        from PIL import Image
                        import cv2

                        cap = cv2.VideoCapture(str(clip_path))
                        cap.set(cv2.CAP_PROP_POS_MSEC, 2500.0)
                        ok, frame = cap.read()
                        cap.release()
                        if ok:
                            images = [Image.fromarray(frame[:, :, ::-1])]
                    elif self.movie_video:
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
                    results_by_id[rec["clip_id"]] = data

            for rec in iter_progress(records, desc="s6 apply", unit="clip"):
                if self.should_skip_clip(rec):
                    continue
                if not rec.get("keep", True) or rec.get("reject"):
                    rec["verified"] = False
                    rec["bucket_verified"] = False
                    rec["confidence"] = 0.0
                    rec["route"] = "other"
                    MetadataManager.mark_done(rec, self.service_id)
                    continue

                data = results_by_id.get(rec["clip_id"], {})
                self._apply_verify_row(rec, data)
                if rec["verified"]:
                    verified_count += 1
                MetadataManager.mark_done(rec, self.service_id)

            self.metadata.write_all(records)
        finally:
            if gemma is not None:
                gemma.cleanup()
            shutdown_ray(self.config)

        return {
            "verified_clips": verified_count,
            "model": gcfg.get("model_name", "Gemma"),
            "gpus": gpu_ids,
            "mode": mode,
            "ray": bool(results_by_id) and self._use_ray(),
        }
