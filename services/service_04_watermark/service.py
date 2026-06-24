"""Service 4: per-clip on-screen text detection via Gemma4 + vLLM."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from common.base_service import BaseService
from common.metadata_manager import MetadataManager
from common.progress import iter_progress
from common.vllm_text_detect import detect_text_clips_vllm


class WatermarkService(BaseService):
    service_id = "s4"
    service_name = "s4_text_detect"
    owned_fields = ["has_text", "text_overlay"]

    def process_movie(self) -> Dict[str, Any]:
        records = self.metadata.read_all()
        to_run = [r for r in records if not self.should_skip_clip(r)]
        text_present = 0
        text_absent = 0
        text_types: Dict[str, int] = {}
        updated = 0

        s4_cfg = self.config.get("pipeline", {}).get("s4", {})
        drop_text = bool(s4_cfg.get("drop_text_clips", False))
        dropped = 0

        if to_run:
            detected = detect_text_clips_vllm(
                self.config,
                to_run,
                movie_dir=self.movie_dir,
                movie_video=self.movie_video,
            )
            for rec in iter_progress(to_run, desc="s4 apply", unit="clip"):
                payload = detected.get(rec["clip_id"], {})
                rec.pop("watermark", None)
                rec["has_text"] = bool(payload.get("has_text"))
                rec["text_overlay"] = payload.get(
                    "text_overlay",
                    {"present": False, "text_type": "none", "confidence": 0.0},
                )
                # Drop clips that contain on-screen text: reject so s5-s12 skip
                # them and they never enter the dataset.
                if drop_text and rec["has_text"]:
                    rec["keep"] = False
                    rec["reject"] = True
                    rec["reject_reason"] = "has_text"
                    dropped += 1
                MetadataManager.mark_done(rec, self.service_id)
                updated += 1

        for rec in records:
            if rec.get("has_text"):
                text_present += 1
                ttype = str((rec.get("text_overlay") or {}).get("text_type") or "other")
                text_types[ttype] = text_types.get(ttype, 0) + 1
            else:
                text_absent += 1

        self.metadata.write_all(records)

        summary = {
            "clips_updated": updated,
            "text_present": text_present,
            "text_absent": text_absent,
            "text_dropped": dropped,
            "drop_text_clips": drop_text,
            "text_types": text_types,
            "model": s4_cfg.get("model_path") or "gemma-4-31b-it",
            "backend": "vllm",
        }
        (self.movie_dir / "movie_text_detect.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary
