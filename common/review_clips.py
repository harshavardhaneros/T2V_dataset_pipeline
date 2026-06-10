"""Link clip MP4s under export/ for local HTTP review (no parent-dir URLs)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def link_clip_under_export(export_dir: Path, clip_src: Path, *, subdir: str = "clips") -> Path:
    """Place clip under export_dir so python -m http.server can serve it."""
    dest_dir = export_dir / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / clip_src.name
    if dest.is_symlink() or dest.exists():
        try:
            if dest.is_symlink():
                target = dest.resolve()
                if target.exists() and target.stat().st_size > 0:
                    return dest
            elif dest.stat().st_size > 0:
                return dest
        except OSError:
            pass
        dest.unlink(missing_ok=True)
    if not clip_src.exists() or clip_src.stat().st_size == 0:
        return dest
    try:
        dest.symlink_to(clip_src.resolve())
    except OSError:
        shutil.copy2(clip_src, dest)
    return dest


def clip_url(html_file: Path, clip_path: Path) -> str:
    """Relative URL from HTML file to clip; must stay inside the serve root."""
    rel = Path(os.path.relpath(clip_path, html_file.parent)).as_posix()
    if rel.startswith(".."):
        raise ValueError(f"clip URL escapes export dir: {rel} ({clip_path})")
    return rel


def link_clips_for_review(export_dir: Path, clips_dir: Path, clip_ids: list[str]) -> Path:
    """Symlink workspace clips into export/clips/ for clip_review.html."""
    out = export_dir / "clips"
    for clip_id in clip_ids:
        src = clips_dir / f"{clip_id}.mp4"
        if src.exists():
            link_clip_under_export(export_dir, src)
    return out


CLIP_SEARCH_DIRS = ("clips", "static_clips", "excessive_motion", "low_quality", "dups")


def find_workspace_clip(workspace: Path, clip_id: str) -> Path | None:
    """Locate clip MP4 in workspace (main or s2 reject folders)."""
    for sub in CLIP_SEARCH_DIRS:
        p = workspace / sub / f"{clip_id}.mp4"
        if p.exists():
            return p
    export_clip = workspace / "export" / "clips" / f"{clip_id}.mp4"
    if export_clip.exists():
        return export_clip
    by_bucket = workspace / "export" / "by_bucket"
    if by_bucket.exists():
        for clip in by_bucket.glob(f"*/clips/{clip_id}.mp4"):
            if clip.exists():
                return clip
    return None


def link_workspace_clips(export_dir: Path, workspace: Path, clip_ids: list[str]) -> None:
    """Symlink all known clip locations into export/clips/."""
    for clip_id in clip_ids:
        src = find_workspace_clip(workspace, clip_id)
        if src:
            link_clip_under_export(export_dir, src)


def link_frames_for_review(export_dir: Path, workspace: Path, clip_ids: list[str]) -> None:
    """Symlink actor frames into export/frames/ for HTTP review."""
    frames_src = workspace / "frames"
    if not frames_src.exists():
        return
    dest_dir = export_dir / "frames"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for clip_id in clip_ids:
        for pattern in (f"{clip_id}.1.jpg", f"{clip_id}.2.jpg", f"{clip_id}.3.jpg", f"{clip_id}.jpg"):
            src = frames_src / pattern
            if not src.exists():
                continue
            dest = dest_dir / pattern
            if dest.exists() or dest.is_symlink():
                continue
            try:
                dest.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dest)
