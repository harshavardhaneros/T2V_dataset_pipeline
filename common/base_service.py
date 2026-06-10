"""Base class for all pipeline services."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from common.metadata_manager import MetadataManager
from common.paths import service_log_dir
from common.progress import service_banner
from common.runtime_tracker import RuntimeTracker
from common.video_files import find_movie_video


class BaseService(ABC):
    service_id: str = ""
    service_name: str = ""
    owned_fields: List[str] = []

    def __init__(
        self,
        movie_dir: Path,
        config: Dict[str, Any],
        pipeline_root: Path,
        force: bool = False,
    ):
        self.movie_dir = Path(movie_dir)
        self.config = config
        self.pipeline_root = Path(pipeline_root)
        self.force = force
        self.metadata = MetadataManager(self.movie_dir)
        self.movie_video = self._resolve_movie_video()
        self.movie_name = self.movie_video.name if self.movie_video else "unknown.mp4"

    def _resolve_movie_video(self) -> Optional[Path]:
        return find_movie_video(self.movie_dir)

    def _log_dir(self) -> Path:
        return service_log_dir(self.config, self.service_id)

    def should_skip_clip(self, record: Dict[str, Any]) -> bool:
        if self.force:
            return False
        return MetadataManager.is_done(record, self.service_id)

    def should_skip_movie(self, records: List[Dict[str, Any]]) -> bool:
        if self.force or not records:
            return False
        return all(self.should_skip_clip(r) for r in records)

    def process_clip(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Override for per-clip logic; default no-op."""
        return record

    @abstractmethod
    def process_movie(self) -> Dict[str, Any]:
        """Run service on entire movie; return stats dict."""
        ...

    def write_metadata(self, records: List[Dict[str, Any]]) -> None:
        for rec in records:
            if not self.should_skip_clip(rec) or self.force:
                MetadataManager.mark_done(rec, self.service_id)
        self.metadata.write_all(records)

    def write_runtime(
        self,
        tracker: RuntimeTracker,
        status: str,
        runtime_seconds: float,
        stats: Dict[str, Any],
    ) -> Path:
        return tracker.write_runtime_log(
            status=status,
            runtime_seconds=runtime_seconds,
            extra_stats=stats,
        )

    def run(self) -> Dict[str, Any]:
        tracker = RuntimeTracker(
            self.service_id,
            self.service_name,
            self.movie_name,
            self._log_dir(),
        )
        tracker.start_timer()
        service_banner(self.service_id, self.service_name)
        try:
            stats = self.process_movie()
            elapsed = tracker.stop_timer()
            self.write_runtime(tracker, "success", elapsed, stats)
            return {"status": "success", "runtime_seconds": elapsed, "stats": stats}
        except Exception as exc:
            elapsed = tracker.stop_timer()
            self.write_runtime(
                tracker, "error", elapsed, {"error": str(exc)}
            )
            raise


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_path_on_syspath(pipeline_root: Path) -> None:
    root = str(pipeline_root)
    if root not in sys.path:
        sys.path.insert(0, root)
