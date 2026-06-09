"""Service 12: aggregate movie report."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from common.base_service import BaseService
from common.metadata_manager import MetadataManager
from common.paths import reports_dir


class ReportService(BaseService):
    service_id = "s12"
    service_name = "s12_report"
    owned_fields = []

    def process_movie(self) -> Dict[str, Any]:
        records = self.metadata.read_all()
        video_id = records[0]["video_id"] if records else self.movie_dir.name

        dup_removed = sum(1 for r in records if not r.get("keep", True))
        rejected = sum(1 for r in records if r.get("reject"))
        final = sum(1 for r in records if r.get("verdict") == "FINAL")
        review = sum(1 for r in records if r.get("verdict") == "REVIEW")
        discard = sum(1 for r in records if r.get("verdict") == "DISCARD")
        bucket_dist = Counter(r.get("bucket") for r in records if r.get("keep", True))
        actor_dist = Counter(r.get("actor_status") for r in records)
        scores = [float(r.get("final_score", 0)) for r in records if r.get("keep", True)]
        total_duration = sum(
            float(r.get("duration", 0))
            for r in records
            if r.get("verdict") in ("FINAL", "REVIEW")
        )

        report = {
            "video_id": video_id,
            "movies_processed": 1,
            "clips_processed": len(records),
            "duplicates_removed": dup_removed,
            "rejected_clips": rejected,
            "final_clips": final,
            "review_clips": review,
            "discarded_clips": discard,
            "bucket_distribution": dict(bucket_dist),
            "actor_distribution": dict(actor_dist),
            "score_distribution": {
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
                "mean": sum(scores) / len(scores) if scores else 0,
            },
            "total_export_duration_sec": round(total_duration, 2),
        }

        rdir = reports_dir(self.config)
        rdir.mkdir(parents=True, exist_ok=True)

        txt_path = rdir / f"{video_id}_report.txt"
        json_path = rdir / f"{video_id}_report.json"

        lines = [f"Report for {video_id}", "=" * 40]
        for k, v in report.items():
            lines.append(f"{k}: {v}")
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        for rec in records:
            if not self.should_skip_clip(rec):
                MetadataManager.mark_done(rec, self.service_id)
        self.metadata.write_all(records)

        return report
