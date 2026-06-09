"""Per-service runtime logging."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class RuntimeTracker:
    def __init__(self, service_id: str, service_name: str, movie_name: str, log_dir: Path):
        self.service_id = service_id
        self.service_name = service_name
        self.movie_name = movie_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._start: Optional[float] = None
        self._stats: Dict[str, Any] = {}

    def start_timer(self) -> None:
        self._start = time.time()

    def stop_timer(self) -> float:
        if self._start is None:
            return 0.0
        elapsed = time.time() - self._start
        self._start = None
        return elapsed

    def add_stat(self, key: str, value: Any) -> None:
        self._stats[key] = value

    def write_runtime_log(
        self,
        status: str = "success",
        runtime_seconds: Optional[float] = None,
        extra_stats: Optional[Dict[str, Any]] = None,
    ) -> Path:
        if runtime_seconds is None:
            runtime_seconds = self.stop_timer() if self._start is None else self.stop_timer()
        stats = {**self._stats, **(extra_stats or {})}
        video_stem = Path(self.movie_name).stem
        out_path = self.log_dir / f"{video_stem}_runtime.json"
        payload = {
            "service": self.service_name,
            "movie": self.movie_name,
            "runtime_seconds": round(runtime_seconds, 2),
            "status": status,
            "stats": stats,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out_path


def write_pipeline_runtime_json(
    out_path: Path,
    *,
    video_id: str,
    movie_name: str,
    movie_dir: Path,
    services: Dict[str, Dict[str, Any]],
    wall_runtime_seconds: float,
    from_step: Optional[str] = None,
    to_step: Optional[str] = None,
    status: str = "success",
    error: Optional[str] = None,
) -> Path:
    """Write one JSON with per-service and total runtime for a movie run."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    service_entries: Dict[str, Any] = {}
    total_service_seconds = 0.0
    for step_id, result in services.items():
        runtime = float(result.get("runtime_seconds", 0) or 0)
        total_service_seconds += runtime
        service_entries[step_id] = {
            "service_name": result.get("service_name", step_id),
            "runtime_seconds": round(runtime, 2),
            "status": result.get("status", "unknown"),
            "stats": result.get("stats", {}),
        }

    payload: Dict[str, Any] = {
        "video_id": video_id,
        "movie": movie_name,
        "workspace": str(movie_dir),
        "from_step": from_step,
        "to_step": to_step,
        "status": status,
        "wall_runtime_seconds": round(wall_runtime_seconds, 2),
        "total_service_runtime_seconds": round(total_service_seconds, 2),
        "services": service_entries,
    }
    if error:
        payload["error"] = error

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def append_runtime_summary(
    summary_path: Path,
    movie: str,
    service_times: Dict[str, float],
) -> None:
    """Append or update runtime_summary.csv row for a movie."""
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [f"s{i}" for i in range(1, 13)] + ["total"]
    header = "movie," + ",".join(cols)
    total = sum(service_times.get(f"s{i}", 0) for i in range(1, 13))
    row_vals = [str(round(service_times.get(f"s{i}", 0), 2)) for i in range(1, 13)]
    row_vals.append(str(round(total, 2)))
    row = movie + "," + ",".join(row_vals) + "\n"

    existing: Dict[str, str] = {}
    if summary_path.exists():
        lines = summary_path.read_text(encoding="utf-8").strip().split("\n")
        if lines:
            for line in lines[1:]:
                if line.strip():
                    parts = line.split(",", 1)
                    if parts:
                        existing[parts[0]] = line
    existing[movie] = row.rstrip()
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for m in sorted(existing.keys()):
            f.write(existing[m] + "\n")
