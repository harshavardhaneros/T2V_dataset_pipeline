"""Service 2: dedup, motion filtering, and DOVER quality filtering."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.base_service import BaseService
from common.clip_io import export_clip_mp4
from common.dedup_bktree import BKTree, phash_to_int
from common.metadata_manager import MetadataManager
from common.motion_filter import (
    classify_motion_failure,
    combine_motion_scores,
    motion_bounds_for_source,
)
from common.motion_unimatch import compute_unimatch_motion
from common.motion_vmaf import compute_vmaf_motion
from common.video_time import clip_local_range
from model_clients.dover_client import DoverClient


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

    def _run_dedup(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        hamming_threshold = int(
            self.config.get("thresholds", {}).get("dedup", {}).get("hamming_threshold", 8)
        )
        tree = BKTree(threshold=hamming_threshold)
        exact_seen: Dict[str, str] = {}
        duplicates = 0
        kept = 0
        reject_dirs = self._reject_dirs()

        for rec in records:
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

        for rec in records:
            if self.should_skip_clip(rec):
                continue
            start, end = clip_local_range(rec, self.config)
            crop_box = rec.get("crop_box", "")
            unimatch = compute_unimatch_motion(
                str(self.movie_video), start, end, model_cfg, crop_box=crop_box
            )
            vmaf = compute_vmaf_motion(
                str(self.movie_video), start, end, model_cfg, crop_box=crop_box
            )
            rec["unimatch_motion"] = round(unimatch, 4)
            rec["vmaf_motion"] = round(vmaf, 4)
            rec["motion_score"] = round(combine_motion_scores(unimatch, vmaf, model_cfg), 4)
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
        dover = DoverClient(self.config)
        reject_dirs = self._reject_dirs()
        passed = rejected = 0

        for rec in records:
            if self.should_skip_clip(rec):
                continue
            if rec.get("motion_score") is None:
                continue

            clip_path: Optional[Path] = None
            if self.movie_video:
                clip_path = self.movie_dir / "clips" / f"{rec['clip_id']}.mp4"
                clip_path.parent.mkdir(parents=True, exist_ok=True)
                if not clip_path.exists():
                    export_clip_mp4(self.movie_video, rec, clip_path)

            scores = dover.score_video(str(clip_path or self.movie_video))
            rec["aesthetic_score"] = round(float(scores["aesthetic_score"]), 4)
            rec["technical_score"] = round(float(scores["technical_score"]), 4)
            rec["dover_score"] = round(float(scores["dover_score"]), 4)

            below = rec["dover_score"] < dover_threshold
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
