"""File-based lock for metadata.jsonl updates (multi-worker safe)."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path


LOCK_NAME = "metadata.lock"
DEFAULT_TIMEOUT = 300
POLL_INTERVAL = 0.1


@contextmanager
def metadata_lock(movie_dir: Path, timeout: float = DEFAULT_TIMEOUT):
    """Acquire metadata.lock in movie_dir, release on exit."""
    movie_dir = Path(movie_dir)
    lock_path = movie_dir / LOCK_NAME
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(POLL_INTERVAL)
    else:
        raise TimeoutError(f"Could not acquire lock: {lock_path}")
    try:
        yield lock_path
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass
