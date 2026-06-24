"""Service 2: dedup, motion filtering, and DOVER quality filtering."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from common.base_service import BaseService
from common.progress import iter_progress, ray_get_progress
from common.clip_io import export_clip_mp4
from common.clip_workers import (
    dover_score_job,
    export_clip_mp4_job,
    score_clip_motion,
)
from common.ray_pool import init_ray, parallel_map, ray_enabled, ray_settings
from common.dedup_bktree import BKTree, phash_to_int
from common.metadata_manager import MetadataManager
from common.motion_filter import (
    classify_motion_failure,
    motion_bounds_for_source,
)
class DedupService(BaseService):
    service_id = "s2"
    service_name = "s2_filter"
    owned_fields = [
        "keep",
        "dup_of",
        "unimatch_motion",
        "vmaf_motion",
        "motion_score",
        "motion_pass",
        "aesthetic_score",
        "technical_score",
        "dover_score",
        "s2_reject_reason",
    ]

    def _score_only(self) -> bool:
        s2_cfg = self.config.get("pipeline", {}).get("s2", {})
        mode = s2_cfg.get("mode", "filter")
        return mode == "score_only"

    def _reject_dirs(self) -> Dict[str, Path]:
        return {
            "duplicate": self.movie_dir / "dups",
            "static": self.movie_dir / "static_clips",
            "excessive_motion": self.movie_dir / "excessive_motion",
            "low_quality": self.movie_dir / "low_quality",
        }

    def _export_rejected_clip(
        self,
        reason: str,
        record: Dict[str, Any],
        reject_dirs: Dict[str, Path],
    ) -> None:
        if not self.movie_video:
            return
        out_dir = reject_dirs.get(reason)
        if out_dir is None:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{record['clip_id']}.mp4"
        if out_path.exists():
            return
        export_clip_mp4(self.movie_video, record, out_path)

    def _motion_workers(self, model_cfg: Dict[str, Any]) -> int:
        rc = ray_settings(self.config)
        uni_dev = str(model_cfg.get("unimatch", {}).get("device", "cpu"))
        if "cuda" in uni_dev:
            return max(1, int(rc.get("motion_workers", rc.get("gpu_workers", 1))))
        return max(1, int(rc.get("motion_workers", rc.get("num_cpus", 8))))

    def _dover_workers(self) -> int:
        rc = ray_settings(self.config)
        return max(1, int(rc.get("dover_workers", rc.get("gpu_workers", 1))))

    def _use_motion_gpu_actors(self) -> bool:
        rc = ray_settings(self.config)
        return bool(rc.get("parallel_motion_gpu", True)) and ray_enabled(self.config)

    def _use_dover_gpu_actors(self) -> bool:
        rc = ray_settings(self.config)
        return bool(rc.get("parallel_dover_gpu", True)) and ray_enabled(self.config)

    def _motion_ray(
        self,
        jobs: List[Dict[str, Any]],
        n_workers: int,
    ) -> Dict[str, Dict[str, Any]]:
        from common.vlm_ray_actors import MotionScoreActor
        from common.gpu_actor_pool import gpu_actor_options

        if MotionScoreActor is None or not init_ray(self.config):
            return {}

        import ray

        n_actors = max(1, min(n_workers, len(jobs)))
        opts = gpu_actor_options(self.config)
        actors = [MotionScoreActor.options(**opts).remote() for _ in range(n_actors)]
        futures = [
            actors[i % n_actors].score.remote(job) for i, job in enumerate(jobs)
        ]
        try:
            rows = ray_get_progress(futures, desc="s2 motion")
        finally:
            for actor in actors:
                try:
                    ray.kill(actor)
                except Exception:
                    pass
        return {row["clip_id"]: row for row in rows}

    def _dover_ray(
        self,
        jobs: List[Dict[str, Any]],
        n_workers: int,
    ) -> List[Dict[str, Any]]:
        from common.vlm_ray_actors import DoverScoreActor
        from common.gpu_actor_pool import gpu_actor_options

        if DoverScoreActor is None or not init_ray(self.config):
            return []

        import ray

        n_actors = max(1, min(n_workers, len(jobs)))
        opts = gpu_actor_options(self.config)
        actors = [DoverScoreActor.options(**opts).remote(self.config) for _ in range(n_actors)]
        futures = [
            actors[i % n_actors].score.remote(job) for i, job in enumerate(jobs)
        ]
        try:
            return ray_get_progress(futures, desc="s2 DOVER")
        finally:
            for actor in actors:
                try:
                    ray.kill(actor)
                except Exception:
                    pass

    def _parallel_export_clips(
        self,
        records: List[Dict[str, Any]],
        clips_dir: Path,
        *,
        label: str,
    ) -> None:
        if not self.movie_video:
            return
        clips_dir.mkdir(parents=True, exist_ok=True)
        rc = ray_settings(self.config)
        parallel_export = bool(rc.get("parallel_clip_export", ray_enabled(self.config)))
        min_parallel = int(rc.get("parallel_clip_min", 4))
        export_cfg = self.config.get("pipeline", {}).get("export", {})
        thresholds = self.config.get("thresholds", {})

        if parallel_export and len(records) >= min_parallel:
            jobs = [
                {
                    "record": rec,
                    "video_path": str(self.movie_video),
                    "clip_path": str(clips_dir / f"{rec['clip_id']}.mp4"),
                    "export_cfg": export_cfg,
                    "thresholds": thresholds,
                }
                for rec in records
            ]
            parallel_map(self.config, export_clip_mp4_job, jobs, label=label)
            return

        for rec in iter_progress(records, desc=f"{label} export", unit="clip"):
            clip_path = clips_dir / f"{rec['clip_id']}.mp4"
            if clip_path.exists() and clip_path.stat().st_size > 0:
                continue
            export_clip_mp4(
                self.movie_video,
                rec,
                clip_path,
                export_cfg=export_cfg,
                thresholds=thresholds,
            )

    def _run_dedup(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        hamming_threshold = int(
            self.config.get("thresholds", {}).get("dedup", {}).get("hamming_threshold", 8)
        )
        tree = BKTree(threshold=hamming_threshold)
        exact_seen: Dict[str, str] = {}
        duplicates = 0
        kept = 0
        reject_dirs = self._reject_dirs()

        for rec in iter_progress(records, desc="s2 dedup", unit="clip"):
            if self.should_skip_clip(rec):
                continue
            phash = rec.get("phash", "")
            clip_id = rec["clip_id"]
            rec["keep"] = True
            rec["dup_of"] = None
            rec["s2_reject_reason"] = None

            if not phash:
                kept += 1
                continue

            if phash in exact_seen:
                rec["dup_of"] = exact_seen[phash]
                rec["s2_reject_reason"] = "duplicate"
                duplicates += 1
                if not self._score_only():
                    rec["keep"] = False
                    self._export_rejected_clip("duplicate", rec, reject_dirs)
            else:
                val = phash_to_int(phash)
                match = tree.find_match(val)
                if match:
                    rec["dup_of"] = match
                    rec["s2_reject_reason"] = "duplicate"
                    duplicates += 1
                    if not self._score_only():
                        rec["keep"] = False
                        self._export_rejected_clip("duplicate", rec, reject_dirs)
                else:
                    exact_seen[phash] = clip_id
                    tree.add(val, clip_id)
                    kept += 1

        return {"kept": kept, "duplicates_removed": duplicates}

    def _run_motion(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        if not self.movie_video:
            raise FileNotFoundError(f"No movie video in {self.movie_dir}")

        motion_cfg = self.config.get("thresholds", {}).get("motion", {})
        model_cfg = {
            "unimatch": self.config.get("models", {}).get("unimatch", {}),
            "vmaf": self.config.get("models", {}).get("vmaf", {}),
            "motion": motion_cfg,
        }
        by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        motion_targets = [rec for rec in records if not self.should_skip_clip(rec)]
        clips_dir = self.movie_dir / "clips"
        self._parallel_export_clips(motion_targets, clips_dir, label="s2_clip_export_motion")

        rc = ray_settings(self.config)
        min_parallel = int(rc.get("parallel_clip_min", 4))
        parallel_motion = bool(rc.get("parallel_motion", True))
        motion_workers = self._motion_workers(model_cfg)

        jobs = [
            {
                "record": rec,
                "video_path": str(self.movie_video),
                "clip_path": str(clips_dir / f"{rec['clip_id']}.mp4"),
                "config": self.config,
                "model_cfg": model_cfg,
            }
            for rec in motion_targets
        ]

        scores_by_id: Dict[str, Dict[str, Any]] = {}
        if parallel_motion and len(motion_targets) >= min_parallel:
            if self._use_motion_gpu_actors():
                scores_by_id = self._motion_ray(jobs, motion_workers)
            if not scores_by_id:
                scores_by_id = {
                    row["clip_id"]: row
                    for row in parallel_map(
                        self.config,
                        score_clip_motion,
                        jobs,
                        label="s2_motion",
                        workers=motion_workers,
                        min_items=min_parallel,
                    )
                }
            for rec in motion_targets:
                row = scores_by_id.get(rec["clip_id"], {})
                rec["unimatch_motion"] = row.get("unimatch_motion", 0.0)
                rec["vmaf_motion"] = row.get("vmaf_motion", 0.0)
                rec["motion_score"] = row.get("motion_score", 0.0)
                source = rec.get("source_video") or rec.get("video_id", "unknown")
                by_source[source].append(rec)
        else:
            for job in iter_progress(jobs, desc="s2 motion", unit="clip"):
                rec = job["record"]
                row = score_clip_motion(job)
                rec["unimatch_motion"] = row.get("unimatch_motion", 0.0)
                rec["vmaf_motion"] = row.get("vmaf_motion", 0.0)
                rec["motion_score"] = row.get("motion_score", 0.0)
                source = rec.get("source_video") or rec.get("video_id", "unknown")
                by_source[source].append(rec)

        static = excessive = passed = 0
        reject_dirs = self._reject_dirs()
        for _source, source_records in by_source.items():
            scores = [float(r["motion_score"]) for r in source_records]
            lower, upper = motion_bounds_for_source(scores, {"motion": motion_cfg})
            for rec in source_records:
                failure = classify_motion_failure(float(rec["motion_score"]), lower, upper)
                if failure:
                    rec["motion_pass"] = False
                    if not rec.get("s2_reject_reason"):
                        rec["s2_reject_reason"] = failure
                    if failure == "static":
                        static += 1
                    else:
                        excessive += 1
                    if not self._score_only():
                        rec["keep"] = False
                        self._export_rejected_clip(failure, rec, reject_dirs)
                else:
                    rec["motion_pass"] = True
                    passed += 1

        return {
            "motion_passed": passed,
            "static_rejected": static,
            "excessive_motion_rejected": excessive,
        }

    def _run_dover(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        dover_threshold = float(
            self.config.get("thresholds", {}).get("dover", {}).get("min_overall_score", 0.60)
        )
        reject_dirs = self._reject_dirs()
        passed = rejected = 0
        rc = ray_settings(self.config)
        parallel_dover = bool(rc.get("parallel_dover", ray_enabled(self.config)))
        min_parallel = int(rc.get("parallel_clip_min", 4))
        dover_workers = self._dover_workers()

        dover_targets = [
            rec
            for rec in records
            if not self.should_skip_clip(rec) and rec.get("motion_score") is not None
        ]
        clips_dir = self.movie_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        jobs = [
            {
                "record": rec,
                "clip_path": str(clips_dir / f"{rec['clip_id']}.mp4"),
                "config": self.config,
            }
            for rec in dover_targets
        ]

        score_rows: List[Dict[str, Any]] = []
        if parallel_dover and len(dover_targets) >= min_parallel:
            if self._use_dover_gpu_actors():
                score_rows = self._dover_ray(jobs, dover_workers)
            if not score_rows:
                score_rows = parallel_map(
                    self.config,
                    dover_score_job,
                    jobs,
                    label="s2_dover",
                    workers=dover_workers,
                    min_items=min_parallel,
                )
        else:
            score_rows = [
                dover_score_job(job)
                for job in iter_progress(jobs, desc="s2 DOVER", unit="clip")
            ]

        scores_by_id = {row["clip_id"]: row for row in score_rows}

        for rec in iter_progress(dover_targets, desc="s2 DOVER apply", unit="clip"):
            clip_path = clips_dir / f"{rec['clip_id']}.mp4"
            if not clip_path.exists() or clip_path.stat().st_size == 0:
                if self.movie_video:
                    export_clip_mp4(self.movie_video, rec, clip_path)

            row = scores_by_id.get(rec["clip_id"], {})
            rec["aesthetic_score"] = row.get("aesthetic_score", 0.0)
            rec["technical_score"] = row.get("technical_score", 0.0)
            rec["dover_score"] = row.get("dover_score", 0.0)

            below = float(rec["dover_score"]) < dover_threshold
            if below:
                if not rec.get("s2_reject_reason"):
                    rec["s2_reject_reason"] = "low_quality"
                rejected += 1
                if not self._score_only():
                    rec["keep"] = False
                    self._export_rejected_clip("low_quality", rec, reject_dirs)
            elif rec.get("motion_pass", False):
                passed += 1

        return {"dover_passed": passed, "low_quality_rejected": rejected}

    def process_movie(self) -> Dict[str, Any]:
        records = self.metadata.read_all()
        if self.should_skip_movie(records) and records:
            return {"skipped": True, "clips": len(records)}

        dedup_stats = self._run_dedup(records)
        motion_stats = self._run_motion(records)
        dover_stats = self._run_dover(records)

        survivors = sum(1 for rec in records if rec.get("keep", True))
        for rec in records:
            if not self.should_skip_clip(rec):
                MetadataManager.mark_done(rec, self.service_id)

        self.metadata.write_all(records)
        return {
            **dedup_stats,
            **motion_stats,
            **dover_stats,
            "survivors": survivors,
            "total": len(records),
            "mode": "score_only" if self._score_only() else "filter",
        }
