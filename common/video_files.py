"""Shared video file extension helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

VIDEO_GLOBS = ("*.mp4", "*.mkv", "*.webm", "*.mov", "*.avi")


def find_movie_video(directory: Path, globs: Iterable[str] = VIDEO_GLOBS) -> Optional[Path]:
    """Return first video file found in directory."""
    directory = Path(directory)
    for pattern in globs:
        for path in sorted(directory.glob(pattern)):
            if path.is_file() or path.is_symlink():
                return path
    return None
