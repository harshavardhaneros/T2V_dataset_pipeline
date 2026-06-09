"""Service 4: movie-level watermark detection (placeholder consensus)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from common.base_service import BaseService
from common.frame_sampler import sample_keyframes
from common.metadata_manager import MetadataManager


class WatermarkService(BaseService):
    service_id = "s4"
    service_name = "s4_watermark"
    owned_fields = ["watermark"]

    def process_movie(self) -> Dict[str, Any]:
        records = self.metadata.read_all()
        meta_path = self.movie_dir / "movie_watermark.json"
        if not self.force and meta_path.exists():
            wm = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            wm = self._detect_watermark(records)
            meta_path.write_text(json.dumps(wm, indent=2), encoding="utf-8")

        updated = 0
        for rec in records:
            if self.should_skip_clip(rec):
                continue
            rec["watermark"] = dict(wm)
            MetadataManager.mark_done(rec, self.service_id)
            updated += 1
        self.metadata.write_all(records)
        return {"clips_updated": updated, "watermark": wm}

    def _detect_watermark(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        sample_n = int(
            self.config.get("thresholds", {}).get("watermark", {}).get("sample_clips", 5)
        )
        active = [r for r in records if r.get("keep", True)][:sample_n]
        corners = ["top_left", "top_right", "bottom_left", "bottom_right"]
        votes: Dict[str, int] = {c: 0 for c in corners}

        if self.movie_video and active:
            for rec in active:
                frames = sample_keyframes(
                    str(self.movie_video),
                    rec["timestamp_start"],
                    rec["timestamp_end"],
                    crop_box=rec.get("crop_box", ""),
                )
                if not frames:
                    continue
                frame = frames[0]
                h, w = frame.shape[:2]
                regions = {
                    "top_left": frame[0 : h // 5, 0 : w // 5],
                    "top_right": frame[0 : h // 5, -w // 5 :],
                    "bottom_left": frame[-h // 5 :, 0 : w // 5],
                    "bottom_right": frame[-h // 5 :, -w // 5 :],
                }
                import numpy as np

                for corner, patch in regions.items():
                    if float(np.std(patch)) > 40:
                        votes[corner] += 1

        best = max(votes, key=votes.get) if votes else "top_right"
        present = votes.get(best, 0) >= 2
        return {
            "present": present,
            "corner": best if present else None,
            "bbox": [0, 0, 80, 40] if present else [],
            "mask_stored": False,
        }
