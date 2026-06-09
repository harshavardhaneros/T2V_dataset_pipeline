"""JSONL metadata read/write with schema versioning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from common.metadata_lock import metadata_lock

SCHEMA_VERSION = "1.0"
METADATA_FILENAME = "metadata.jsonl"


def new_clip_record(
    video_id: str,
    clip_id: str,
    scene_id: int,
    source_video: str,
    timestamp_start: float,
    timestamp_end: float,
    duration: float,
    phash: str = "",
    crop_box: str = "",
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "clip_id": clip_id,
        "scene_id": scene_id,
        "source_video": source_video,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "duration": duration,
        "phash": phash,
        "crop_box": crop_box,
        "processing_status": {f"done_s{i}": False for i in range(1, 13)},
    }


class MetadataManager:
    def __init__(self, movie_dir: Path):
        self.movie_dir = Path(movie_dir)
        self.path = self.movie_dir / METADATA_FILENAME

    def exists(self) -> bool:
        return self.path.exists()

    def read_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def write_all(self, records: List[Dict[str, Any]], use_lock: bool = True) -> None:
        self.movie_dir.mkdir(parents=True, exist_ok=True)

        def _write():
            with open(self.path, "w", encoding="utf-8") as f:
                for rec in records:
                    rec.setdefault("schema_version", SCHEMA_VERSION)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if use_lock:
            with metadata_lock(self.movie_dir):
                _write()
        else:
            _write()

    def update_records(
        self,
        updater: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        with metadata_lock(self.movie_dir):
            records = self.read_all()
            records = updater(records)
            with open(self.path, "w", encoding="utf-8") as f:
                for rec in records:
                    rec.setdefault("schema_version", SCHEMA_VERSION)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return records

    @staticmethod
    def mark_done(record: Dict[str, Any], step: str) -> None:
        ps = record.setdefault("processing_status", {})
        ps[f"done_{step}"] = True

    @staticmethod
    def is_done(record: Dict[str, Any], step: str) -> bool:
        return record.get("processing_status", {}).get(f"done_{step}", False)
