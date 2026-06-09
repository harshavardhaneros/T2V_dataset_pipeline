"""Bridge to Master_Pipeline_t2i_dataset actor_tagger + captioner."""

from __future__ import annotations

import importlib
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import cv2

logger = logging.getLogger(__name__)

_MASTER_ROOT: Optional[Path] = None
_imported = False
_SAVED_MODULES: Dict[str, Any] = {}


def init_master(master_root: str | Path) -> Path:
    """Register Master_Pipeline root (imports use master_import_context)."""
    global _MASTER_ROOT, _imported
    root = Path(master_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Master pipeline root not found: {root}")
    _MASTER_ROOT = root
    _imported = True
    return root


@contextmanager
def master_import_context() -> Iterator[None]:
    """Temporarily prioritize Master_Pipeline modules over indic `common` package."""
    root = str(master_root())
    indic_root = str(Path(__file__).resolve().parent.parent)
    saved_path = sys.path[:]
    saved_modules = {}
    for name in (
        "common",
        "actor_tagger",
        "captioner",
        "vlm_backend",
        "qwen_vl_utils",
    ):
        if name in sys.modules:
            saved_modules[name] = sys.modules.pop(name)

    sys.path = [p for p in sys.path if p != indic_root]
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        yield
    finally:
        for name, mod in saved_modules.items():
            sys.modules[name] = mod
        sys.path = saved_path


def master_root() -> Path:
    if _MASTER_ROOT is None:
        raise RuntimeError("Call init_master() before using master bridge")
    return _MASTER_ROOT


def resolve_master_path(rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return master_root() / p


# bucket_XX -> captioner canonical category (captioner.BUCKET_PROMPT_FILES keys)
BUCKET_ID_TO_CATEGORY = {
    "bucket_01": "people_portraits",
    "bucket_02": "clothing_textiles",
    "bucket_03": "architecture",
    "bucket_04": "landscape_nature",
    "bucket_05": "urban_street",
    "bucket_06": "rural_village",
    "bucket_07": "food_drink",
    "bucket_08": "festivals_rituals",
    "bucket_09": "objects_artifacts",
    "bucket_10": "animals_wildlife",
    "bucket_11": "art_design",
    "bucket_12": "abstract_texture",
}


def bucket_to_category(bucket_id: str, slug: str = "") -> str:
    if bucket_id in BUCKET_ID_TO_CATEGORY:
        return BUCKET_ID_TO_CATEGORY[bucket_id]
    with master_import_context():
        from captioner import normalize_bucket

        return normalize_bucket(slug or bucket_id)


def ensure_yolo_face_model(yolo_path: Path) -> Path:
    """Download yolov12n-face.pt if missing (akanametov/yolo-face release)."""
    yolo_path = Path(yolo_path)
    if yolo_path.exists():
        return yolo_path
    yolo_path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/"
        "yolov12n-face.pt"
    )
    logger.info("Downloading YOLO face model -> %s", yolo_path)
    try:
        import urllib.request

        urllib.request.urlretrieve(url, yolo_path)
    except Exception as exc:
        raise FileNotFoundError(
            f"YOLO face model missing at {yolo_path}. Download manually:\n  {url}\n"
            f"Error: {exc}"
        ) from exc
    return yolo_path


def save_clip_keyframe(
    video_path: Path,
    record: Dict[str, Any],
    frames_dir: Path,
) -> Optional[Path]:
    """Save middle frame as actor_frames/{clip_id}.jpg."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    out = frames_dir / f"{record['clip_id']}.jpg"
    if out.exists():
        return out

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    from common.video_time import clip_local_middle

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    mid = clip_local_middle(record)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    crop_box = record.get("crop_box", "")
    if crop_box:
        from common.ffmpeg_utils import parse_crop_box

        crop = parse_crop_box(crop_box)
        if crop:
            cw, ch, cx, cy = crop
            frame = frame[cy : cy + ch, cx : cx + cw]

    cv2.imwrite(str(out), frame)
    return out


def tag_actor_frames(
    image_paths: List[Path],
    cfg: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run Master actor_tagger.tag_frames on extracted keyframes."""
    yolo_rel = cfg.get("yolo_face_model", "yolov12n-face.pt")
    yolo_path = Path(yolo_rel)
    if not yolo_path.is_absolute():
        models_root = Path(cfg.get("models_root", ""))
        if models_root and (models_root / yolo_rel).exists():
            yolo_path = models_root / yolo_rel
        else:
            yolo_path = resolve_master_path(f"actors/{yolo_rel}")
    yolo_path = ensure_yolo_face_model(yolo_path)
    embeddings_dir = resolve_master_path(
        cfg.get("actor_embeddings_dir", "actors/actor_embeddings")
    )
    gpu_id = int(cfg.get("actor_tag_gpu_id", 0))
    threshold = float(cfg.get("actor_similarity_threshold", 0.35))

    with master_import_context():
        from actor_tagger import tag_frames

        return tag_frames(
            [str(p) for p in image_paths],
            actor_embeddings_dir=embeddings_dir,
            output_dir=output_dir,
            yolo_model_path=yolo_path,
            gpu_id=gpu_id,
            similarity_threshold=threshold,
        )


def load_master_prompts(prompt_dir: Path) -> Dict[str, str]:
    with master_import_context():
        from captioner import load_prompts

        return load_prompts(prompt_dir)


def build_caption_prompt(bucket_prompt: str, actors: List[Dict[str, Any]]) -> str:
    """Actor-aware caption prompt (removes 'do not name actors' rule when tagged)."""
    from common.actor_caption import build_actor_caption_prompt

    return build_actor_caption_prompt(bucket_prompt, actors)


def parse_caption_output(raw: str) -> Dict[str, Any]:
    with master_import_context():
        from captioner import parse_caption_json

        return parse_caption_json(raw)


def create_vlm_backend(cfg: Dict[str, Any]):
    with master_import_context():
        from vlm_backend import create_backend

        backend = create_backend(
            cfg.get("caption_backend", "transformers"),
            model_path=cfg.get("model_path"),
            gpu_ids=[int(g) for g in cfg.get("caption_gpu_ids", [0])],
            max_new_tokens=int(cfg.get("max_new_tokens", 512)),
        )
        backend.load()
        return backend


def caption_image(vlm_backend, image_path: Path, prompt: str) -> str:
    from PIL import Image

    pil = Image.open(image_path).convert("RGB")
    max_side = 1024
    w, h = pil.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return vlm_backend.generate(pil, prompt)
