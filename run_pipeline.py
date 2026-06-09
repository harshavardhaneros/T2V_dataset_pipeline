#!/usr/bin/env python3
"""Indic video dataset pipeline runner."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.base_service import ensure_path_on_syspath, load_yaml
from common.paths import logs_dir, outputs_root, reports_dir, service_log_dir, workspaces_dir
from common.run_output import init_run_output, run_output_enabled
from common.video_files import find_movie_video
from common.prompt_manager import PromptManager
from common.runtime_tracker import append_runtime_summary, write_pipeline_runtime_json
from common.service_registry import get_service_class


def pipeline_root() -> Path:
    return Path(__file__).resolve().parent


def load_config(root: Path, pipeline_yaml: str = "pipeline.yaml") -> Dict[str, Any]:
    pipeline_path = root / "configs" / pipeline_yaml
    if not pipeline_path.name.endswith(".yaml"):
        pipeline_path = root / "configs" / f"{pipeline_yaml}.yaml"
    pipeline = load_yaml(pipeline_path)
    models = load_yaml(root / "configs" / "models.yaml")
    for section, overrides in (pipeline.get("models_overrides") or {}).items():
        if isinstance(overrides, dict):
            base = models.get(section, {})
            if isinstance(base, dict):
                models[section] = {**base, **overrides}
            else:
                models[section] = overrides
    cfg = {
        "pipeline": pipeline,
        "thresholds": load_yaml(root / "configs" / "thresholds.yaml"),
        "models": models,
    }
    mp = pipeline.get("master_pipeline", {})
    if mp.get("root"):
        from common.master_bridge import init_master

        init_master(mp["root"])
    return cfg


def setup_movie_workspace(
    movie_path: Path,
    workspaces_path: Path,
    video_id: Optional[str] = None,
    *,
    isolated_run: bool = False,
) -> Path:
    movie_path = Path(movie_path).resolve()
    if not movie_path.exists():
        raise FileNotFoundError(movie_path)
    vid = video_id or movie_path.stem
    if isolated_run:
        movie_dir = workspaces_path
    else:
        movie_dir = workspaces_path / vid
    movie_dir.mkdir(parents=True, exist_ok=True)
    dest = movie_dir / movie_path.name
    if not dest.exists():
        try:
            dest.symlink_to(movie_path)
        except OSError:
            shutil.copy2(movie_path, dest)
    elif dest.resolve() != movie_path.resolve() and not dest.is_symlink():
        pass
    return movie_dir


def run_services_for_movie(
    movie_dir: Path,
    config: Dict[str, Any],
    root: Path,
    from_step: Optional[str],
    to_step: Optional[str],
    force: bool,
    service_order: List[str],
) -> Dict[str, float]:
    prompt_zip = config["pipeline"].get("prompt_pack", {}).get("zip_path")
    if prompt_zip:
        config["_prompt_manager"] = PromptManager(prompt_zip)

    start_idx = 0
    if from_step:
        if from_step not in service_order:
            raise ValueError(f"Unknown step: {from_step}")
        start_idx = service_order.index(from_step)

    end_idx = len(service_order)
    if to_step:
        if to_step not in service_order:
            raise ValueError(f"Unknown step: {to_step}")
        end_idx = service_order.index(to_step) + 1

    timings: Dict[str, float] = {}
    service_results: Dict[str, Dict[str, Any]] = {}
    pipeline_status = "success"
    pipeline_error: Optional[str] = None
    ran_from = service_order[start_idx] if start_idx < len(service_order) else None
    ran_to = service_order[end_idx - 1] if end_idx > 0 else None
    wall_start = time.time()
    movie_video = find_movie_video(movie_dir)
    movie_name = movie_video.name if movie_video else f"{movie_dir.name}.mp4"

    try:
        for step_id in service_order[start_idx:end_idx]:
            cls = get_service_class(step_id)
            svc = cls(movie_dir, config, root, force=force)
            movie_name = svc.movie_name
            print(f"[{movie_dir.name}] Running {svc.service_name}...")
            try:
                result = svc.run()
            except Exception as exc:
                elapsed = 0.0
                log_path = (
                    service_log_dir(config, step_id)
                    / f"{Path(svc.movie_name).stem}_runtime.json"
                )
                if log_path.exists():
                    elapsed = float(
                        json.loads(log_path.read_text(encoding="utf-8")).get(
                            "runtime_seconds", 0
                        )
                    )
                service_results[step_id] = {
                    "service_name": svc.service_name,
                    "runtime_seconds": elapsed,
                    "status": "error",
                    "stats": {"error": str(exc)},
                }
                timings[step_id] = elapsed
                pipeline_status = "error"
                pipeline_error = str(exc)
                raise

            timings[step_id] = result.get("runtime_seconds", 0)
            service_results[step_id] = {
                "service_name": svc.service_name,
                "runtime_seconds": timings[step_id],
                "status": result.get("status", "success"),
                "stats": result.get("stats", {}),
            }
            print(f"  done in {timings[step_id]:.2f}s — {result.get('stats', {})}")
    finally:
        wall_elapsed = time.time() - wall_start
        runtime_json = reports_dir(config) / f"{movie_dir.name}_pipeline_runtime.json"
        write_pipeline_runtime_json(
            runtime_json,
            video_id=movie_dir.name,
            movie_name=movie_name,
            movie_dir=movie_dir,
            services=service_results,
            wall_runtime_seconds=wall_elapsed,
            from_step=ran_from,
            to_step=ran_to,
            status=pipeline_status,
            error=pipeline_error,
        )
        print(f"Pipeline runtime JSON: {runtime_json}")

        append_runtime_summary(
            reports_dir(config) / "runtime_summary.csv",
            movie_dir.name,
            timings,
        )

    return timings


def load_registry(registry_path: Path) -> List[Dict[str, str]]:
    rows = []
    with open(registry_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main() -> int:
    root = pipeline_root()
    ensure_path_on_syspath(root)
    parser = argparse.ArgumentParser(description="Indic Video Dataset Pipeline")
    parser.add_argument("--config", type=str, default="pipeline.yaml", help="Config under configs/ e.g. pipeline_v2.yaml")
    parser.add_argument("--movie", type=str, help="Path to a single movie file")
    parser.add_argument("--movie-registry", type=str, help="CSV registry for batch run")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (registry mode)")
    parser.add_argument("--from-step", type=str, default=None, help="Resume from service id e.g. s5")
    parser.add_argument("--force", action="store_true", help="Re-run all steps")
    parser.add_argument("--video-id", type=str, default=None, help="Override workspace folder name")
    parser.add_argument(
        "--to-step",
        type=str,
        default=None,
        help="Stop after this service (inclusive), e.g. s7 for actor-tagging-only test",
    )
    parser.add_argument(
        "--time-offset",
        type=float,
        default=0.0,
        help="Seconds added to clip timestamps (use when input is a trimmed segment)",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Test mode: cap clips at s1 extract and limit s8 caption count",
    )
    parser.add_argument(
        "--outputs-root",
        type=str,
        default=None,
        help="Override pipeline outputs root (e.g. .../v2_outputs)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Resume into an existing isolated run directory",
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Force a new isolated run folder (default for full pipeline when isolate_run_outputs=true)",
    )
    args, _unknown = parser.parse_known_args()

    config = load_config(root, args.config)
    if args.outputs_root:
        config["pipeline"]["outputs_root"] = args.outputs_root
    out = outputs_root(config)
    if not run_output_enabled(config):
        for i in range(1, 13):
            (logs_dir(config) / f"s{i}").mkdir(parents=True, exist_ok=True)
        reports_dir(config).mkdir(parents=True, exist_ok=True)
        workspaces_dir(config).mkdir(parents=True, exist_ok=True)
    print(f"Outputs root: {out}")
    if run_output_enabled(config):
        layout = config["pipeline"].get("output_layout", "per_movie")
        print(f"Output layout: {layout} (v2_outputs/{{video_id}}/ per movie)")

    if args.max_clips or args.time_offset:
        config["_test"] = {
            "max_clips": args.max_clips,
            "time_offset_sec": float(args.time_offset or 0),
        }
        if args.max_clips:
            print(f"[test] max_clips={args.max_clips}, time_offset={args.time_offset}s")

    service_order = config["pipeline"].get("services_order", [f"s{i}" for i in range(1, 13)])

    if not args.movie and not args.movie_registry:
        parser.error("Provide --movie or --movie-registry")

    def _prepare_run(movie_path: Path, video_id: Optional[str]) -> Path:
        vid = video_id or movie_path.stem
        isolated = run_output_enabled(config)
        if not isolated:
            ws_path = workspaces_dir(config)
            ws_path.mkdir(parents=True, exist_ok=True)
            return setup_movie_workspace(
                movie_path, ws_path, vid, isolated_run=False
            )

        if args.run_dir:
            init_run_output(
                config,
                video_id=vid,
                source_movie=movie_path,
                run_dir=Path(args.run_dir),
                new_run=False,
            )
        elif config["pipeline"].get("output_layout") == "per_movie":
            init_run_output(
                config,
                video_id=vid,
                source_movie=movie_path,
                new_run=False,
            )
        elif args.new_run or not args.from_step or args.from_step == "s1":
            init_run_output(
                config,
                video_id=vid,
                source_movie=movie_path,
                new_run=True,
            )
        else:
            init_run_output(
                config,
                video_id=vid,
                source_movie=movie_path,
                new_run=False,
            )

        run_root = Path(config["_run"]["root"])
        print(f"Movie output: {run_root}")
        return setup_movie_workspace(
            movie_path,
            workspaces_dir(config),
            vid,
            isolated_run=True,
        )

    if args.movie:
        movie_path = Path(args.movie)
        movie_dir = _prepare_run(movie_path, args.video_id)
        run_services_for_movie(
            movie_dir, config, root, args.from_step, args.to_step, args.force, service_order
        )
        if config.get("_run"):
            print(f"Pipeline complete. Output: {config['_run']['root']}")
        else:
            print(f"Pipeline complete: {movie_dir}")
        return 0

    registry_path = root / args.movie_registry if not Path(args.movie_registry).is_absolute() else Path(args.movie_registry)
    rows = load_registry(registry_path)
    pending = [r for r in rows if r.get("status", "pending") == "pending"]

    def _run_row(row: Dict[str, str]) -> None:
        src = Path(row["source_path"])
        vid = row.get("video_id", src.stem)
        if run_output_enabled(config):
            init_run_output(
                config,
                video_id=vid,
                source_movie=src,
                new_run=config["pipeline"].get("output_layout") != "per_movie",
            )
            print(f"Movie output: {config['_run']['root']}")
        mdir = setup_movie_workspace(
            src,
            workspaces_dir(config),
            vid,
            isolated_run=run_output_enabled(config),
        )
        run_services_for_movie(
            mdir, config, root, args.from_step, args.to_step, args.force, service_order
        )

    if args.workers <= 1:
        for row in pending:
            print(f"Processing {row.get('video_id')}...")
            _run_row(row)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_run_row, row): row for row in pending}
            for fut in as_completed(futures):
                row = futures[fut]
                try:
                    fut.result()
                    print(f"Finished {row.get('video_id')}")
                except Exception as exc:
                    print(f"Failed {row.get('video_id')}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
