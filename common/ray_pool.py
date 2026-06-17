"""Optional Ray parallelization with ProcessPool / sequential fallback."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

_PIPELINE_ROOT: Optional[str] = None


def ray_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("pipeline", {}).get("ray", {}) or {}


def ray_enabled(config: Dict[str, Any]) -> bool:
    return bool(ray_settings(config).get("enabled", False))


def _pipeline_root(config: Dict[str, Any]) -> str:
    root = config.get("pipeline", {}).get("pipeline_root", "")
    if root:
        return str(root)
    return str(os.environ.get("INDIC_PIPELINE_ROOT", ""))


def _worker_init(pipeline_root: str) -> None:
    import sys

    global _PIPELINE_ROOT
    _PIPELINE_ROOT = pipeline_root
    if pipeline_root and pipeline_root not in sys.path:
        sys.path.insert(0, pipeline_root)


def init_ray(config: Dict[str, Any]) -> bool:
    """Start Ray if enabled and installed. Returns True when Ray is ready."""
    if not ray_enabled(config):
        return False
    try:
        import ray
    except ImportError:
        logger.warning("ray is not installed; falling back to ProcessPoolExecutor")
        return False

    if ray.is_initialized():
        return True

    rc = ray_settings(config)
    root = _pipeline_root(config)
    init_kwargs: Dict[str, Any] = {
        "ignore_reinit_error": True,
        "logging_level": logging.ERROR,
    }
    if rc.get("num_cpus"):
        init_kwargs["num_cpus"] = int(rc["num_cpus"])
    try:
        import torch

        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            if rc.get("num_gpus"):
                init_kwargs["num_gpus"] = int(rc["num_gpus"])
            else:
                init_kwargs["num_gpus"] = n_gpu
    except ImportError:
        pass
    env = dict(os.environ)
    if root:
        env["INDIC_PIPELINE_ROOT"] = root
        env["PYTHONPATH"] = root + (
            ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
    init_kwargs["runtime_env"] = {"env_vars": env}
    ray.init(**init_kwargs)
    return True


def shutdown_ray(config: Dict[str, Any] | None = None) -> None:
    """Release Ray GPUs and restore CUDA_VISIBLE_DEVICES for later vLLM steps."""
    if config is not None and not ray_enabled(config):
        return
    from common.qwen_vllm import shutdown_ray_after_service

    shutdown_ray_after_service()


def parallel_map(
    config: Dict[str, Any],
    func: Callable[[T], R],
    items: List[T],
    *,
    label: str = "tasks",
    workers: Optional[int] = None,
    min_items: Optional[int] = None,
) -> List[R]:
    """Run func on each item in parallel when worthwhile (Ray or processes)."""
    if not items:
        return []

    rc = ray_settings(config)
    clip_min = int(min_items if min_items is not None else rc.get("parallel_clip_min", 4))
    if len(items) < clip_min:
        return [func(item) for item in items]

    chunk_size = int(rc.get("chunk_size", 64))
    if workers is None:
        workers = int(rc.get("num_workers") or rc.get("num_cpus") or (os.cpu_count() or 4))
    workers = max(1, min(workers, len(items)))

    if init_ray(config):
        import ray

        from common.progress import progress_enabled, ray_get_progress

        remote = ray.remote(func)
        futures = [remote.remote(item) for item in items]
        if progress_enabled() and len(futures) > 1:
            results = ray_get_progress(futures, desc=label)
        else:
            results = list(ray.get(futures))
        logger.info("Ray parallel_map %s: %d items", label, len(items))
        return results

    root = _pipeline_root(config)
    logger.info(
        "ProcessPool parallel_map %s: %d items, %d workers",
        label,
        len(items),
        workers,
    )
    from common.progress import iter_progress, progress_enabled

    chunksize = max(1, len(items) // (workers * 4))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(root,),
    ) as pool:
        if progress_enabled() and len(items) > 1:
            return list(
                iter_progress(
                    pool.map(func, items, chunksize=chunksize),
                    desc=label,
                    total=len(items),
                )
            )
        return list(pool.map(func, items, chunksize=chunksize))
