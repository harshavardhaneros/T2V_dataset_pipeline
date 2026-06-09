"""Service 5: Qwen3-VL-32B classification — 3 keyframes + majority vote."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from common.base_service import BaseService
from common.gpu_info import log_service_gpus
from common.metadata_manager import MetadataManager
from common.paths import qwen_model_path
from common.vlm_service import (
    QwenVLMService,
    clip_keyframe_images,
    majority_vote_buckets,
    parse_vlm_json,
)


CLASSIFY_JSON_PROMPT = """Classify this video frame into ONE of 12 buckets.
Reject if: black frames, credits, ads, title cards, corruption, excessive text, blank, unrelated.

Buckets: bucket_01 people_portraits, bucket_02 clothing_textiles, bucket_03 architecture,
bucket_04 landscape_nature, bucket_05 urban_street, bucket_06 rural_village, bucket_07 food_drink,
bucket_08 festivals_rituals, bucket_09 objects_artifacts, bucket_10 animals_wildlife,
bucket_11 art_design, bucket_12 abstract_texture.

Return ONLY JSON:
{"bucket":"bucket_01","bucket_confidence":0.9,"reject":false,"reject_reason":null}
"""


def _normalize_bucket(raw: str, valid: List[str]) -> str:
    raw = (raw or "").strip().lower()
    if raw in valid:
        return raw
    for v in valid:
        if raw.replace("_", "") in v.replace("_", ""):
            return v
    mapping = {
        "people_portraits": "bucket_01",
        "clothing_textiles": "bucket_02",
        "architecture": "bucket_03",
        "landscape_nature": "bucket_04",
        "urban_street": "bucket_05",
        "rural_village": "bucket_06",
        "food_drink": "bucket_07",
        "festivals_rituals": "bucket_08",
        "objects_artifacts": "bucket_09",
        "animals_wildlife": "bucket_10",
        "art_design": "bucket_11",
        "abstract_texture": "bucket_12",
    }
    return mapping.get(raw, valid[0] if valid else "bucket_01")


class ClassifyService(BaseService):
    service_id = "s5"
    service_name = "s5_classify"
    owned_fields = ["bucket", "bucket_confidence", "reject", "reject_reason"]

    def process_movie(self) -> Dict[str, Any]:
        records = self.metadata.read_all()
        mp = self.config["pipeline"]["master_pipeline"]
        gpu_ids = [int(g) for g in mp.get("classify_gpu_ids", [0, 1, 2, 3])]
        log_service_gpus(
            "s5",
            "VLM classify — 3-frame majority vote",
            f"{mp.get('vlm_model_name', 'Qwen3-VL-32B')} @ {qwen_model_path(self.config)}",
            gpu_ids,
        )

        prompt_mgr = self.config.get("_prompt_manager")
        valid_buckets = prompt_mgr.bucket_ids if prompt_mgr else [f"bucket_{i:02d}" for i in range(1, 13)]

        vlm = QwenVLMService.acquire(self.config, gpu_ids, "s5")
        classified = rejected = 0

        try:
            for rec in records:
                if self.should_skip_clip(rec):
                    continue
                if not rec.get("keep", True):
                    MetadataManager.mark_done(rec, self.service_id)
                    continue

                images = []
                if self.movie_video:
                    images = clip_keyframe_images(
                        self.movie_video, rec, self.config, [0.25, 0.5, 0.75]
                    )

                bucket_votes: List[str] = []
                reject_votes = 0
                confidences: List[float] = []

                if not images:
                    rec["bucket"] = valid_buckets[0]
                    rec["bucket_confidence"] = 0.0
                    rec["reject"] = True
                    rec["reject_reason"] = "no_keyframes"
                    rejected += 1
                else:
                    for img in images:
                        raw = vlm.generate(img, CLASSIFY_JSON_PROMPT)
                        data = parse_vlm_json(raw, {})
                        bucket_votes.append(
                            _normalize_bucket(data.get("bucket", ""), valid_buckets)
                        )
                        if data.get("reject"):
                            reject_votes += 1
                        confidences.append(float(data.get("bucket_confidence", 0.5)))

                    if reject_votes >= 2:
                        rec["reject"] = True
                        rec["reject_reason"] = "majority_reject"
                        rec["bucket"] = bucket_votes[0] if bucket_votes else valid_buckets[0]
                        rec["bucket_confidence"] = 0.0
                        rejected += 1
                    else:
                        bucket, vote_conf = majority_vote_buckets(bucket_votes)
                        rec["bucket"] = _normalize_bucket(bucket, valid_buckets)
                        rec["bucket_confidence"] = round(
                            sum(confidences) / len(confidences) * vote_conf, 4
                        )
                        rec["reject"] = False
                        rec["reject_reason"] = None

                classified += 1
                MetadataManager.mark_done(rec, self.service_id)

            self.metadata.write_all(records)
        finally:
            QwenVLMService.release()

        return {
            "classified": classified,
            "rejected": rejected,
            "model": mp.get("vlm_model_name", "Qwen3-VL-32B"),
            "gpus": gpu_ids,
            "vote_frames": 3,
        }
