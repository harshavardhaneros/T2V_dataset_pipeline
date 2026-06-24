"""Service 11: export captions, clips, and per-bucket manifests (eros-style)."""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from common.base_service import BaseService
from common.progress import iter_progress
from common.caption_text import prose_caption_for_export
from common.clip_io import export_clip_mp4
from common.clip_workers import export_clip_mp4_job
from common.ray_pool import parallel_map, ray_enabled, ray_settings
from common.review_clips import link_clip_under_export
from common.metadata_manager import MetadataManager
from common.video_files import find_movie_video


class ExportService(BaseService):
    service_id = "s11"
    service_name = "s11_export"
    owned_fields = []

    def process_movie(self) -> Dict[str, Any]:
        records = self.metadata.read_all()
        export_cfg = self.config.get("pipeline", {}).get("export", {})
        allowed = set(export_cfg.get("include_verdicts", ["FINAL", "REVIEW"]))

        export_dir = self.movie_dir / "export"
        clips_dir = self.movie_dir / "clips"
        by_bucket_dir = export_dir / "by_bucket"
        export_dir.mkdir(parents=True, exist_ok=True)
        clips_dir.mkdir(parents=True, exist_ok=True)
        by_bucket_dir.mkdir(parents=True, exist_ok=True)

        captions_jsonl = export_dir / "captions.jsonl"
        captions_csv = export_dir / f"{self.movie_dir.name}_captions.csv"
        metadata_csv = export_dir / "metadata.csv"
        bucket_index_path = export_dir / "bucket_index.json"

        exported: List[Dict[str, Any]] = []
        for rec in records:
            if rec.get("verdict") in allowed and rec.get("keep", True) and not rec.get("reject"):
                exported.append(rec)

        source = self.movie_video or find_movie_video(self.movie_dir)
        clips_written = 0
        if source and export_cfg.get("export_clips", True):
            thresholds = self.config.get("thresholds")
            rc = ray_settings(self.config)
            parallel_export = bool(
                rc.get("parallel_clip_export", ray_enabled(self.config))
            )
            min_parallel = int(rc.get("parallel_clip_min", 4))

            # Existing clips (already cut by s2/s8 via the same export_cfg) are
            # reused by default — the worker skips any clip that already exists.
            # Set export.force_reencode_clips: true to force a fresh re-encode
            # (e.g. after changing export_cfg); combined with --force it drops
            # the old targets first so workers regenerate them in parallel.
            if self.force and bool(export_cfg.get("force_reencode_clips", False)):
                for rec in exported:
                    clip_out = clips_dir / f"{rec['clip_id']}.mp4"
                    if clip_out.exists():
                        try:
                            clip_out.unlink()
                        except OSError:
                            pass

            jobs = [
                {
                    "record": rec,
                    "video_path": str(source),
                    "clip_path": str(clips_dir / f"{rec['clip_id']}.mp4"),
                    "export_cfg": export_cfg,
                    "thresholds": thresholds,
                }
                for rec in exported
            ]
            if parallel_export and len(jobs) >= min_parallel:
                results = parallel_map(
                    self.config,
                    export_clip_mp4_job,
                    jobs,
                    label="s11 export clips",
                    min_items=min_parallel,
                )
            else:
                results = [
                    export_clip_mp4_job(job)
                    for job in iter_progress(
                        jobs, desc="s11 export clips", unit="clip"
                    )
                ]
            clips_written = sum(1 for r in results if r)

        with open(captions_jsonl, "w", encoding="utf-8") as f:
            for rec in exported:
                line = {
                    "clip_id": rec["clip_id"],
                    "video_id": rec["video_id"],
                    "source_video": rec["source_video"],
                    "timestamp_start": rec["timestamp_start"],
                    "timestamp_end": rec["timestamp_end"],
                    "caption": prose_caption_for_export(rec),
                    "bucket": rec.get("bucket", ""),
                    "verdict": rec.get("verdict", ""),
                    "final_score": rec.get("final_score", 0),
                    "clip_actors": rec.get("clip_actors", []),
                    "caption_verify_score": rec.get("caption_verify_score"),
                    "caption_verify_pass": rec.get("caption_verify_pass"),
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        csv_cols = [
            "clip_id", "scene_id", "bucket", "verdict", "final_score",
            "timestamp_start", "timestamp_end", "duration",
            "clip_actors", "actors_f1", "actors_f2", "actors_f3",
            "pos_f1", "pos_f2", "pos_f3",
            "frame1", "frame2", "frame3",
            "caption",
            "caption_verify_score",
            "caption_verify_pass",
            "caption_verify_explanation",
            "clip_mp4",
        ]
        with open(captions_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
            writer.writeheader()
            for rec in exported:
                row = {k: rec.get(k, "") for k in csv_cols}
                row["caption"] = prose_caption_for_export(rec)
                row["clip_mp4"] = str(clips_dir / f"{rec['clip_id']}.mp4")
                writer.writerow(row)

        if exported:
            fieldnames = list(exported[0].keys())
            with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(exported)

        bucket_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in exported:
            bucket_groups[rec.get("bucket", "unknown")].append(rec)

        bucket_index: Dict[str, Any] = {}
        for bucket, recs in sorted(bucket_groups.items()):
            bdir = by_bucket_dir / bucket
            bclips = bdir / "clips"
            bdir.mkdir(parents=True, exist_ok=True)
            bclips.mkdir(parents=True, exist_ok=True)
            manifest = bdir / "manifest.jsonl"
            with open(manifest, "w", encoding="utf-8") as mf:
                for rec in recs:
                    clip_src = clips_dir / f"{rec['clip_id']}.mp4"
                    clip_dst = bclips / f"{rec['clip_id']}.mp4"
                    if clip_src.exists() and not clip_dst.exists():
                        try:
                            clip_dst.symlink_to(clip_src.resolve())
                        except OSError:
                            shutil.copy2(clip_src, clip_dst)
                    entry = {
                        "clip_id": rec["clip_id"],
                        "bucket": bucket,
                        "timestamp_start": rec["timestamp_start"],
                        "timestamp_end": rec["timestamp_end"],
                        "clip_actors": rec.get("clip_actors", []),
                        "caption": prose_caption_for_export(rec),
                        "clip_mp4": str(clip_dst if clip_dst.exists() else clip_src),
                        "verdict": rec.get("verdict"),
                        "final_score": rec.get("final_score"),
                    }
                    mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            bucket_index[bucket] = {
                "count": len(recs),
                "manifest": str(manifest),
                "clips_dir": str(bclips),
            }

        bucket_index_path.write_text(
            json.dumps(bucket_index, indent=2), encoding="utf-8"
        )

        for rec in exported:
            clip_src = clips_dir / f"{rec['clip_id']}.mp4"
            if clip_src.exists():
                link_clip_under_export(export_dir, clip_src)

        for rec in records:
            if not self.should_skip_clip(rec):
                MetadataManager.mark_done(rec, self.service_id)
        self.metadata.write_all(records)

        return {
            "exported_clips": len(exported),
            "clips_mp4": clips_written,
            "buckets": len(bucket_groups),
            "captions_jsonl": str(captions_jsonl),
            "captions_csv": str(captions_csv),
            "bucket_index": str(bucket_index_path),
        }
