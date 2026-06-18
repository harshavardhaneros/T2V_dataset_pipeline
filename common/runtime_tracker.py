"""Per-service runtime logging."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.paths import reports_dir, service_log_dir


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


def _service_step_ids(num_services: int = 12) -> List[str]:
    return [f"s{i}" for i in range(1, num_services + 1)]


def _read_service_runtime_log(log_path: Path) -> Optional[Dict[str, Any]]:
    if not log_path.exists():
        return None
    try:
        return json.loads(log_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def collect_service_timings_from_logs(
    config: Dict[str, Any],
    movie_stem: str,
    *,
    num_services: int = 12,
) -> Dict[str, float]:
    """Load per-step runtime_seconds from logs/sN/{movie_stem}_runtime.json."""
    timings: Dict[str, float] = {}
    for step_id in _service_step_ids(num_services):
        log_path = service_log_dir(config, step_id) / f"{movie_stem}_runtime.json"
        payload = _read_service_runtime_log(log_path)
        if payload is not None:
            timings[step_id] = float(payload.get("runtime_seconds", 0) or 0)
    return timings


def collect_service_results_from_logs(
    config: Dict[str, Any],
    movie_stem: str,
    *,
    num_services: int = 12,
) -> Dict[str, Dict[str, Any]]:
    """Load per-step service payloads for pipeline runtime JSON."""
    results: Dict[str, Dict[str, Any]] = {}
    for step_id in _service_step_ids(num_services):
        log_path = service_log_dir(config, step_id) / f"{movie_stem}_runtime.json"
        payload = _read_service_runtime_log(log_path)
        if payload is None:
            continue
        results[step_id] = {
            "service_name": payload.get("service", step_id),
            "runtime_seconds": float(payload.get("runtime_seconds", 0) or 0),
            "status": payload.get("status", "success"),
            "stats": payload.get("stats", {}),
        }
    return results


def merge_service_timings(
    log_timings: Dict[str, float],
    run_timings: Dict[str, float],
) -> Dict[str, float]:
    """Prefer current-run timings; keep log timings for steps not run this phase."""
    merged = dict(log_timings)
    for step_id, runtime in run_timings.items():
        merged[step_id] = float(runtime or 0)
    return merged


def merge_service_results(
    log_results: Dict[str, Dict[str, Any]],
    run_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Prefer current-run service results; keep log results for steps not run this phase."""
    merged = dict(log_results)
    merged.update(run_results)
    return merged


def merge_runtime_summary_row(
    existing_vals: Dict[str, float],
    new_vals: Dict[str, float],
) -> Dict[str, float]:
    """Merge CSV row values: keep existing non-zero when new step was not run (0)."""
    merged = dict(existing_vals)
    for step_id, runtime in new_vals.items():
        if runtime > 0 or step_id not in merged:
            merged[step_id] = runtime
    return merged


def rebuild_runtime_artifacts(
    config: Dict[str, Any],
    *,
    video_id: str,
    movie_name: str,
    movie_dir: Path,
    movie_stem: Optional[str] = None,
    num_services: int = 12,
) -> Dict[str, float]:
    """Rebuild runtime_summary.csv and pipeline_runtime.json from per-service logs."""
    stem = movie_stem or Path(movie_name).stem
    timings = collect_service_timings_from_logs(config, stem, num_services=num_services)
    service_results = collect_service_results_from_logs(
        config, stem, num_services=num_services
    )
    total = sum(timings.get(f"s{i}", 0) for i in range(1, num_services + 1))

    step_ids = [step_id for step_id in _service_step_ids(num_services) if step_id in timings]
    from_step = step_ids[0] if step_ids else None
    to_step = step_ids[-1] if step_ids else None

    runtime_json = reports_dir(config) / f"{video_id}_pipeline_runtime.json"
    write_pipeline_runtime_json(
        runtime_json,
        video_id=video_id,
        movie_name=movie_name,
        movie_dir=movie_dir,
        services=service_results,
        wall_runtime_seconds=total,
        from_step=from_step,
        to_step=to_step,
        status="success",
    )

    append_runtime_summary(
        reports_dir(config) / "runtime_summary.csv",
        video_id,
        timings,
    )
    return timings


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

    existing_rows: Dict[str, str] = {}
    if summary_path.exists():
        lines = summary_path.read_text(encoding="utf-8").strip().split("\n")
        if lines:
            for line in lines[1:]:
                if line.strip():
                    parts = line.split(",", 1)
                    if parts:
                        existing_rows[parts[0]] = line

    merged_times = dict(service_times)
    if movie in existing_rows:
        old_parts = existing_rows[movie].split(",")
        if len(old_parts) >= len(cols) + 1:
            old_vals = {
                f"s{i}": float(old_parts[i] or 0)
                for i in range(1, 13)
            }
            merged_times = merge_runtime_summary_row(old_vals, service_times)

    total = sum(merged_times.get(f"s{i}", 0) for i in range(1, 13))
    row_vals = [str(round(merged_times.get(f"s{i}", 0), 2)) for i in range(1, 13)]
    row_vals.append(str(round(total, 2)))
    row = movie + "," + ",".join(row_vals)

    existing_rows[movie] = row
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for m in sorted(existing_rows.keys()):
            f.write(existing_rows[m] + "\n")
