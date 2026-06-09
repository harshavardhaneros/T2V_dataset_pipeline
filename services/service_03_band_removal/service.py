"""Service 3: lazy band removal flag (no re-encode)."""

from __future__ import annotations

from typing import Any, Dict, List

from common.base_service import BaseService
from common.metadata_manager import MetadataManager


class BandRemovalService(BaseService):
    service_id = "s3"
    service_name = "s3_band_removal"
    owned_fields = ["band_removed"]

    def process_movie(self) -> Dict[str, Any]:
        records = self.metadata.read_all()
        applied = 0
        for rec in records:
            if self.should_skip_clip(rec):
                continue
            if not rec.get("keep", True):
                MetadataManager.mark_done(rec, self.service_id)
                continue
            rec["band_removed"] = bool(rec.get("crop_box"))
            applied += 1
            MetadataManager.mark_done(rec, self.service_id)
        self.metadata.write_all(records)
        return {"band_removed_marked": applied}
