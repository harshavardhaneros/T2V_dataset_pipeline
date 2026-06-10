"""Service 10: quality gate verdict."""

from __future__ import annotations

from typing import Any, Dict, List

from common.base_service import BaseService
from common.progress import iter_progress
from common.metadata_manager import MetadataManager


class GateService(BaseService):
    service_id = "s10"
    service_name = "s10_gate"
    owned_fields = ["verdict"]

    def process_movie(self) -> Dict[str, Any]:
        records = self.metadata.read_all()
        gate = self.config.get("thresholds", {}).get("gate", {})
        discard_below = float(gate.get("discard_below", 0.10))
        review_below = float(gate.get("review_below", 0.18))

        counts = {"DISCARD": 0, "REVIEW": 0, "FINAL": 0}
        for rec in iter_progress(records, desc="s10 gate", unit="clip"):
            if self.should_skip_clip(rec):
                continue
            if not rec.get("keep", True) or rec.get("reject"):
                rec["verdict"] = "DISCARD"
            else:
                score = float(rec.get("final_score", 0))
                if score < discard_below:
                    rec["verdict"] = "DISCARD"
                elif score < review_below:
                    rec["verdict"] = "REVIEW"
                else:
                    rec["verdict"] = "FINAL"
            counts[rec["verdict"]] = counts.get(rec["verdict"], 0) + 1
            MetadataManager.mark_done(rec, self.service_id)

        self.metadata.write_all(records)
        return counts
