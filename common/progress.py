"""tqdm helpers for pipeline services (disable with PIPELINE_NO_TQDM=1)."""

from __future__ import annotations

import os
import sys
from typing import Any, Iterable, Iterator, List, Optional, Sequence, TypeVar

T = TypeVar("T")


def progress_enabled() -> bool:
    if os.environ.get("PIPELINE_NO_TQDM", "").lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("CI", "").lower() in ("1", "true", "yes"):
        return False
    if not hasattr(sys.stderr, "isatty"):
        return True
    return bool(sys.stderr.isatty())


def service_banner(service_id: str, service_name: str) -> None:
    if not progress_enabled():
        print(f"=== [{service_id}] {service_name} ===", flush=True)
        return
    from tqdm import tqdm

    tqdm.write(f"\n=== [{service_id}] {service_name} ===")


def iter_progress(
    items: Iterable[T],
    *,
    desc: str,
    unit: str = "clip",
    total: Optional[int] = None,
) -> Iterator[T]:
    if not progress_enabled():
        yield from items
        return
    from tqdm import tqdm

    if total is None and hasattr(items, "__len__"):
        try:
            total = len(items)  # type: ignore[arg-type]
        except TypeError:
            total = None
    yield from tqdm(items, desc=desc, unit=unit, total=total, dynamic_ncols=True)


def progress_batched(
    items: Sequence[T],
    batch_size: int,
    *,
    desc: str,
) -> Iterator[List[T]]:
    total = len(items)
    if total == 0:
        return
    if not progress_enabled():
        for start in range(0, total, batch_size):
            yield list(items[start : start + batch_size])
        return
    from tqdm import tqdm

    with tqdm(total=total, desc=desc, unit="clip", dynamic_ncols=True) as bar:
        for start in range(0, total, batch_size):
            batch = list(items[start : start + batch_size])
            yield batch
            bar.update(len(batch))


def ray_get_progress(futures: Sequence[Any], *, desc: str) -> List[Any]:
    """ray.get with a tqdm bar (completion order; callers usually key by clip_id)."""
    if not futures:
        return []
    import ray

    if not progress_enabled() or len(futures) == 1:
        return list(ray.get(list(futures)))

    from tqdm import tqdm

    pending = list(futures)
    results: List[Any] = []
    with tqdm(total=len(pending), desc=desc, unit="clip", dynamic_ncols=True) as bar:
        while pending:
            done, pending = ray.wait(pending, num_returns=1, timeout=5.0)
            if not done:
                continue
            results.extend(ray.get(done))
            bar.update(len(done))
    return results
