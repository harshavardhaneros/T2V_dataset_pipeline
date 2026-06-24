"""Shared video file extension helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
VIDEO_GLOBS = tuple(f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS))


def is_video_file(path: Path) -> bool:
    path = Path(path)
    return path.suffix.lower() in VIDEO_EXTENSIONS and (path.is_file() or path.is_symlink())


def list_movie_videos(directory: Path, globs: Iterable[str] = VIDEO_GLOBS) -> List[Path]:
    """Return all video files in a directory (sorted, de-duplicated by resolved path)."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    seen: set[Path] = set()
    found: List[Path] = []
    for pattern in globs:
        for path in sorted(directory.glob(pattern)):
            if not is_video_file(path):
                continue
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            found.append(path)
    return found


def find_movie_video(directory: Path, globs: Iterable[str] = VIDEO_GLOBS) -> Optional[Path]:
    """Return first video file found in directory."""
    videos = list_movie_videos(directory, globs=globs)
    return videos[0] if videos else None
