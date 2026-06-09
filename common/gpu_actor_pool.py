"""Ray actor pool — one GPU model replica per actor."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, TypeVar

from common.ray_pool import init_ray, ray_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def gpu_actor_count(config: Dict[str, Any], requested: List[int] | None = None) -> int:
    rc = ray_settings(config)
    if rc.get("gpu_workers"):
        return max(1, int(rc["gpu_workers"]))
    if requested:
        from common.gpu_info import resolve_gpu_ids

        return max(1, len(resolve_gpu_ids(requested)))
    try:
        import torch

        return max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
    except ImportError:
        return 1


def parallel_gpu_map(
    config: Dict[str, Any],
    actor_cls: Any,
    method: str,
    items: List[T],
    *,
    label: str = "gpu_tasks",
) -> List[R]:
    """Dispatch items round-robin across GPU Ray actors."""
    if not items:
        return []

    rc = ray_settings(config)
    min_items = int(rc.get("parallel_clip_min", 2))
    if len(items) < min_items:
        # Sequential fallback via a single in-process worker is handled by callers.
        return []

    if not init_ray(config):
        return []

    import ray

    n_actors = gpu_actor_count(config)
    actors = [actor_cls.remote(config) for _ in range(n_actors)]
    logger.info("%s: %d items → %d GPU actors", label, len(items), n_actors)

    futures = []
    for i, item in enumerate(items):
        actor = actors[i % n_actors]
        futures.append(getattr(actor, method).remote(item))

    try:
        return ray.get(futures)
    finally:
        for actor in actors:
            try:
                ray.kill(actor)
            except Exception:
                pass
