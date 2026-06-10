"""Service 5: fast bucket classify (Qwen2.5-VL-7B, 1 frame, optional Ray 2-GPU)."""

from __future__ import annotations

from typing import Any, Dict, List

from common.base_service import BaseService
from common.progress import iter_progress, ray_get_progress
from common.gpu_info import log_service_gpus, resolve_gpu_ids
from common.metadata_manager import MetadataManager
from common.paths import qwen_classify_model_path
from common.qwen_classify import QwenClassifyWorker


class ClassifyService(BaseService):
    service_id = "s5"
    service_name = "s5_classify"
    owned_fields = ["bucket", "bucket_confidence", "reject", "reject_reason"]

    def _s5_cfg(self) -> Dict[str, Any]:
        return self.config.get("pipeline", {}).get("s5", {})

    def _vote_fractions(self) -> List[float]:
        n = int(self._s5_cfg().get("vote_frames", 1))
        if n <= 1:
            return [0.5]
        if n == 2:
            return [0.33, 0.66]
        return [0.25, 0.5, 0.75]

    def _backend(self) -> str:
        return str(self._s5_cfg().get("backend", "transformers")).lower()

    def _use_vllm(self) -> bool:
        return self._backend() == "vllm"

    def _use_ray(self) -> bool:
        if self._use_vllm():
            return False
        rc = self.config.get("pipeline", {}).get("ray", {})
        return bool(self._s5_cfg().get("parallel", rc.get("parallel_gpu_classify", False)))

    def _classify_ray(
        self,
        targets: List[Dict[str, Any]],
        valid_buckets: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        from common.gpu_actor_pool import gpu_actor_count
        from common.ray_pool import init_ray
        from common.vlm_ray_actors import QwenClassifyActor

        if QwenClassifyActor is None or not init_ray(self.config):
            return {}

        import ray

        mp = self.config["pipeline"]["master_pipeline"]
        gpu_ids = [int(g) for g in mp.get("classify_gpu_ids", [0, 1])]
        n_actors = gpu_actor_count(self.config, gpu_ids)
        actors = [QwenClassifyActor.remote(self.config) for _ in range(n_actors)]
        fractions = self._vote_fractions()
        payloads = [
            {
                "record": rec,
                "config": self.config,
                "video_path": str(self.movie_video) if self.movie_video else "",
                "valid_buckets": valid_buckets,
                "fractions": fractions,
            }
            for rec in targets
        ]
        futures = [
            actors[i % n_actors].classify.remote(payload)
            for i, payload in enumerate(payloads)
        ]
        try:
            rows = ray_get_progress(futures, desc="s5 classify")
        finally:
            for actor in actors:
                try:
                    ray.kill(actor)
                except Exception:
                    pass
        return {row["clip_id"]: row for row in rows}

    def process_movie(self) -> Dict[str, Any]:
        records = self.metadata.read_all()
        mp = self.config["pipeline"]["master_pipeline"]
        s5 = self._s5_cfg()
        if self._use_vllm():
            gpu_ids = resolve_gpu_ids(
                [int(g) for g in s5.get("gpu_ids", mp.get("classify_gpu_ids", [0]))]
            )
        else:
            gpu_ids = resolve_gpu_ids([int(g) for g in mp.get("classify_gpu_ids", [0, 1])])
        model_path = str(s5.get("classify_model_path") or qwen_classify_model_path(self.config))

        log_service_gpus(
            "s5",
            f"VLM classify — {s5.get('vote_frames', 1)} frame(s)",
            model_path,
            gpu_ids,
            extra=(
                "vLLM batched"
                if self._use_vllm()
                else ("Ray multi-GPU" if self._use_ray() and len(gpu_ids) > 1 else "")
            ),
        )

        prompt_mgr = self.config.get("_prompt_manager")
        valid_buckets = (
            prompt_mgr.bucket_ids if prompt_mgr
            else [f"bucket_{i:02d}" for i in range(1, 13)]
        )

        targets = [
            rec for rec in records
            if not self.should_skip_clip(rec) and rec.get("keep", True)
        ]

        classified = rejected = 0
        results_by_id: Dict[str, Dict[str, Any]] = {}

        if self._use_vllm() and len(targets) >= 1:
            from common.vllm_classify import classify_clips_vllm

            clips_dir = self.movie_dir / "clips"
            results_by_id = classify_clips_vllm(
                self.config,
                targets,
                valid_buckets,
                video_path=str(self.movie_video) if self.movie_video else "",
                clips_dir=clips_dir,
                fractions=self._vote_fractions(),
            )
        elif self._use_ray() and len(gpu_ids) > 1 and len(targets) >= 2:
            results_by_id = self._classify_ray(targets, valid_buckets)

        worker: QwenClassifyWorker | None = None
        if not results_by_id:
            worker = QwenClassifyWorker(self.config, device=f"cuda:{gpu_ids[0]}")
            fractions = self._vote_fractions()
            for rec in iter_progress(targets, desc="s5 classify", unit="clip"):
                row = worker.classify_clip({
                    "record": rec,
                    "config": self.config,
                    "video_path": str(self.movie_video) if self.movie_video else "",
                    "valid_buckets": valid_buckets,
                    "fractions": fractions,
                })
                results_by_id[rec["clip_id"]] = row

        for rec in iter_progress(records, desc="s5 apply", unit="clip"):
            if self.should_skip_clip(rec):
                continue
            if not rec.get("keep", True):
                MetadataManager.mark_done(rec, self.service_id)
                continue

            row = results_by_id.get(rec["clip_id"])
            if row and not row.get("skipped"):
                rec["bucket"] = row.get("bucket", valid_buckets[0])
                rec["bucket_confidence"] = row.get("bucket_confidence", 0.0)
                rec["reject"] = bool(row.get("reject", False))
                rec["reject_reason"] = row.get("reject_reason")
                if rec["reject"]:
                    rejected += 1
            classified += 1
            MetadataManager.mark_done(rec, self.service_id)

        if worker:
            worker.cleanup()

        self.metadata.write_all(records)
        return {
            "classified": classified,
            "rejected": rejected,
            "model": model_path,
            "gpus": gpu_ids,
            "vote_frames": len(self._vote_fractions()),
            "parallel_ray": bool(results_by_id) and self._use_ray(),
            "backend": self._backend(),
        }
