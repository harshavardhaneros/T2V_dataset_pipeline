#!/usr/bin/env python3
"""
Actor Tagger — face detection + InsightFace embedding + actor assignment.

Wraps the actors/ face recognition logic into a clean single-process module
usable by master_pipeline.py (no Ray dependency, GPU-accelerated).

Pipeline per image:
  1. YOLOv8/v12-face detects face bounding boxes
  2. InsightFace (buffalo_l) extracts 512-d L2-normalised embeddings
  3. Cosine similarity against pre-built actor .pkl embeddings
  4. Above-threshold matches → actor label assigned

One-time setup (builds actor reference embeddings):
    from actor_tagger import build_actor_embeddings
    build_actor_embeddings(
        "actors/actor_images",
        "actors/actor_embeddings",
    )

Per-run tagging:
    from actor_tagger import tag_frames
    results = tag_frames(
        image_paths,
        actor_embeddings_dir="actors/actor_embeddings",
        output_dir=work_dir / "actor_tags",
        yolo_model_path="actors/yolov12n-face.pt",
    )
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import IMAGE_EXTS, unique_stem as _unique_stem

logger = logging.getLogger(__name__)


def _slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:120] or "actor"


def _compute_iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def _normalize_cast_keys(cast_filter: list[str] | None) -> set[str] | None:
    if not cast_filter:
        return None
    return {_slugify(str(name)) for name in cast_filter if str(name).strip()}


def _normalize_gender_map(
    actor_gender_map: dict[str, str] | None,
) -> dict[str, str]:
    if not actor_gender_map:
        return {}
    out: dict[str, str] = {}
    for name, gender in actor_gender_map.items():
        key = _slugify(str(name))
        g = str(gender).strip().lower()
        if key and g in {"male", "female", "m", "f"}:
            out[key] = "male" if g in {"male", "m"} else "female"
    return out


def _face_gender_label(face: Any) -> str | None:
    """InsightFace genderage: 0=female, 1=male."""
    g = getattr(face, "gender", None)
    if g is None:
        return None
    try:
        return "female" if int(g) == 0 else "male"
    except (TypeError, ValueError):
        return None


def _pick_gender_aware_match(
    sims: np.ndarray,
    actor_keys: list[str],
    actor_genders: dict[str, str],
    face_gender: str | None,
) -> tuple[int | None, float, float]:
    order = np.argsort(sims)[::-1]
    accepted: list[int] = []
    for idx in order:
        actor_key = actor_keys[int(idx)]
        expected = actor_genders.get(_slugify(actor_key))
        if face_gender and expected and expected != face_gender:
            continue
        accepted.append(int(idx))
        if len(accepted) >= 2:
            break
    if not accepted:
        return None, 0.0, 0.0
    best_idx = accepted[0]
    best_sim = float(sims[best_idx])
    second_sim = float(sims[accepted[1]]) if len(accepted) > 1 else 0.0
    return best_idx, best_sim, second_sim


def _load_actor_embeddings(
    embeddings_dir: Path,
    cast_filter: list[str] | None = None,
) -> tuple[list[str], np.ndarray, dict[str, str]]:
    """Load actor .pkl files → (actor_keys, matrix [N×512], display_names)."""
    actor_keys: list[str] = []
    embeddings: list[np.ndarray] = []
    display_names: dict[str, str] = {}
    allowed = _normalize_cast_keys(cast_filter)

    for pkl_path in sorted(embeddings_dir.glob("*.pkl")):
        stem_key = _slugify(pkl_path.stem)
        if allowed is not None and stem_key not in allowed:
            continue
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            emb = np.array(data["embedding"], dtype=np.float32).reshape(-1)
            name = data.get("actor_name", pkl_path.stem)
            name_key = _slugify(name)
            if allowed is not None and name_key not in allowed:
                continue
            actor_keys.append(name)
            embeddings.append(emb)
            display_names[name] = name.replace("_", " ").title()
        except Exception as e:
            logger.warning(f"  Could not load {pkl_path.name}: {e}")

    if not embeddings:
        return [], np.empty((0, 512), dtype=np.float32), {}

    matrix = np.vstack(embeddings).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / (norms + 1e-8)
    if allowed is not None:
        logger.info(
            "[ActorTag] Cast filter active — %d actors (from %d requested keys)",
            len(actor_keys),
            len(allowed),
        )
    return actor_keys, matrix, display_names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_actor_embeddings(
    actor_images_dir: str | Path,
    output_dir: str | Path,
    gpu_id: int = 0,
    model_pack: str = "buffalo_l",
    det_size: tuple[int, int] = (640, 640),
    force: bool = False,
) -> dict[str, Path]:
    """Build one averaged .pkl embedding per actor from reference photos.

    Each subdirectory of actor_images_dir is treated as one actor.
    Skips actors whose .pkl already exists unless force=True.

    Returns mapping: actor_name → pkl_path.
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        raise ImportError(
            "insightface not installed. Run: pip install insightface onnxruntime-gpu"
        )
    import torch

    actor_images_dir = Path(actor_images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actor_dirs = sorted(p for p in actor_images_dir.iterdir() if p.is_dir())
    if not actor_dirs:
        raise RuntimeError(f"No actor subdirectories found in {actor_images_dir}")

    use_gpu = torch.cuda.is_available() and gpu_id >= 0
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_gpu
        else ["CPUExecutionProvider"]
    )
    logger.info(f"[ActorEmbed] Initializing InsightFace {model_pack} ...")
    app = FaceAnalysis(name=model_pack, providers=providers)
    app.prepare(ctx_id=0 if use_gpu else -1, det_size=det_size)

    results: dict[str, Path] = {}
    for actor_dir in actor_dirs:
        actor_name = actor_dir.name
        pkl_path = output_dir / f"{_slugify(actor_name)}.pkl"

        if pkl_path.exists() and not force:
            logger.info(f"  SKIP  {actor_name} (already built)")
            results[actor_name] = pkl_path
            continue

        images = sorted(
            p for p in actor_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not images:
            logger.warning(f"  SKIP  {actor_name} — no images")
            continue

        embs: list[np.ndarray] = []
        for img_path in images:
            try:
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue
                faces = app.get(bgr)
                if not faces:
                    continue
                best = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                embs.append(best.normed_embedding.astype(np.float32))
            except Exception as e:
                logger.warning(f"    {actor_name}/{img_path.name}: {e}")

        if not embs:
            logger.warning(f"  SKIP  {actor_name} — no faces detected in any reference image")
            continue

        mean_emb = np.mean(np.stack(embs), axis=0)
        mean_emb = (mean_emb / (np.linalg.norm(mean_emb) + 1e-12)).astype(np.float32)

        payload = {
            "actor_name": actor_name,
            "embedding": mean_emb.tolist(),
            "model": f"insightface_{model_pack}",
            "num_images_used": len(embs),
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f)

        logger.info(
            f"  OK    {actor_name} ({len(embs)}/{len(images)} images used) → {pkl_path.name}"
        )
        results[actor_name] = pkl_path

    logger.info(f"[ActorEmbed] Built {len(results)} actor embeddings → {output_dir}")
    return results


_SESSIONS: dict[str, "ActorTaggerSession"] = {}


def _session_key(
    actor_embeddings_dir: Path,
    yolo_model_path: Path,
    gpu_id: int,
    similarity_threshold: float,
    similarity_margin: float,
    cast_filter: list[str] | None,
    actor_gender_map: dict[str, str] | None,
    yolo_batch_size: int,
    yolo_imgsz: int,
    yolo_conf: float,
) -> str:
    cast_key = ",".join(sorted(_normalize_cast_keys(cast_filter) or []))
    gender_key = ",".join(
        f"{k}:{v}" for k, v in sorted(_normalize_gender_map(actor_gender_map).items())
    )
    return (
        f"{yolo_model_path.resolve()}|{actor_embeddings_dir.resolve()}|"
        f"{gpu_id}|{similarity_threshold}|{similarity_margin}|{cast_key}|{gender_key}|"
        f"{yolo_batch_size}|{yolo_imgsz}|{yolo_conf}"
    )


class ActorTaggerSession:
    """Reusable YOLO + InsightFace session (one load per Ray worker / process)."""

    def __init__(
        self,
        actor_embeddings_dir: Path,
        yolo_model_path: Path,
        gpu_id: int,
        similarity_threshold: float,
        similarity_margin: float = 0.10,
        cast_filter: list[str] | None = None,
        actor_gender_map: dict[str, str] | None = None,
        yolo_batch_size: int = 32,
        yolo_imgsz: int = 640,
        yolo_conf: float = 0.5,
    ):
        try:
            from ultralytics import YOLO
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                f"Missing dependency for actor tagging: {e}\n"
                "Run: pip install ultralytics insightface onnxruntime-gpu"
            )
        import torch

        self.yolo_batch_size = yolo_batch_size
        self.yolo_imgsz = yolo_imgsz
        self.yolo_conf = yolo_conf
        self.similarity_threshold = similarity_threshold
        self.similarity_margin = similarity_margin
        self.actor_genders = _normalize_gender_map(actor_gender_map)

        self.actor_keys, self.actor_matrix, self.display_names = _load_actor_embeddings(
            actor_embeddings_dir, cast_filter=cast_filter
        )
        if not self.actor_keys:
            raise RuntimeError(f"No actor .pkl files in {actor_embeddings_dir}")

        use_gpu = torch.cuda.is_available() and gpu_id >= 0
        self.device = f"cuda:{gpu_id}" if use_gpu else "cpu"

        logger.info(
            f"[ActorTag] Loading models ({len(self.actor_keys)} actors, "
            f"device={self.device}) ..."
        )
        self.yolo = YOLO(str(yolo_model_path))
        self.yolo.to(self.device)
        try:
            self.yolo.fuse()
        except Exception:
            pass

        import onnxruntime as ort

        ort_providers = ort.get_available_providers()
        if use_gpu and "CUDAExecutionProvider" in ort_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            if use_gpu:
                logger.warning(
                    "[ActorTag] onnxruntime-gpu not installed — InsightFace runs on CPU"
                )
            providers = ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=gpu_id if use_gpu else -1, det_size=(640, 640))
        logger.info("[ActorTag] Models ready")

    def tag_frames(self, image_paths: list, output_dir: Path) -> dict[str, list[dict]]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = [Path(p) for p in image_paths]
        if not image_paths:
            return {}

        logger.info(f"[ActorTag] Tagging {len(image_paths)} images")
        all_results: dict[str, list[dict]] = {}
        total_faces = 0
        total_recognised = 0

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None  # type: ignore

        batch_range = range(0, len(image_paths), self.yolo_batch_size)
        if tqdm is not None:
            batch_range = tqdm(  # type: ignore[assignment]
                batch_range,
                desc="ActorTag YOLO",
                unit="batch",
                dynamic_ncols=True,
            )
        for batch_start in batch_range:
            batch = image_paths[batch_start : batch_start + self.yolo_batch_size]

            loaded: dict[Path, np.ndarray] = {}
            for p in batch:
                try:
                    bgr = cv2.imread(str(p))
                    if bgr is not None:
                        loaded[p] = bgr
                except Exception as e:
                    logger.warning(f"  Could not load {p.name}: {e}")

            if not loaded:
                continue

            valid_paths = list(loaded.keys())
            valid_imgs = [loaded[p] for p in valid_paths]

            try:
                yolo_results = self.yolo.predict(
                    source=valid_imgs,
                    verbose=False,
                    device=self.device,
                    imgsz=self.yolo_imgsz,
                    conf=self.yolo_conf,
                    workers=0,
                    half=True,
                )
            except Exception as e:
                logger.warning(f"  YOLO batch error: {e}")
                continue

            for img_path, yolo_res, bgr in zip(valid_paths, yolo_results, valid_imgs):
                if yolo_res.boxes is None or len(yolo_res.boxes) == 0:
                    continue

                xyxy = yolo_res.boxes.xyxy.detach().cpu().numpy()
                confs_arr = yolo_res.boxes.conf.detach().cpu().numpy()

                try:
                    if_faces = self.app.get(bgr)
                except Exception as e:
                    logger.warning(f"  InsightFace error for {img_path.name}: {e}")
                    continue

                if not if_faces:
                    continue

                if_boxes = [tuple(map(float, f.bbox.tolist())) for f in if_faces]
                image_actors: list[dict] = []

                for j in range(len(xyxy)):
                    yolo_box = tuple(map(float, xyxy[j]))
                    total_faces += 1

                    best_iou, best_if_idx = -1.0, -1
                    for k, if_box in enumerate(if_boxes):
                        iou = _compute_iou(yolo_box, if_box)
                        if iou > best_iou:
                            best_iou, best_if_idx = iou, k

                    if best_if_idx < 0:
                        continue

                    emb = (
                        if_faces[best_if_idx]
                        .normed_embedding.astype(np.float32)
                        .reshape(1, -1)
                    )
                    sims = (emb @ self.actor_matrix.T).flatten()
                    if_face = if_faces[best_if_idx]
                    face_gender = _face_gender_label(if_face)
                    best_actor_idx, best_sim, second_sim = _pick_gender_aware_match(
                        sims,
                        self.actor_keys,
                        self.actor_genders,
                        face_gender,
                    )
                    if best_actor_idx is None:
                        continue
                    margin = best_sim - second_sim

                    if best_sim < self.similarity_threshold:
                        continue
                    if margin < self.similarity_margin:
                        continue

                    total_recognised += 1
                    actor_key = self.actor_keys[best_actor_idx]
                    image_actors.append({
                        "actor": actor_key,
                        "display_name": self.display_names.get(
                            actor_key, actor_key.replace("_", " ").title()
                        ),
                        "similarity": round(best_sim, 4),
                        "similarity_margin": round(margin, 4),
                        "face_gender": face_gender,
                        "bbox": [
                            int(yolo_box[0]), int(yolo_box[1]),
                            int(yolo_box[2]), int(yolo_box[3]),
                        ],
                        "yolo_confidence": round(float(confs_arr[j]), 4),
                    })

                if image_actors:
                    all_results[str(img_path)] = image_actors
                    ustem = _unique_stem(img_path)
                    out_path = output_dir / f"{ustem}_actors.json"
                    with open(out_path, "w") as f:
                        json.dump({
                            "image": str(img_path),
                            "image_name": img_path.name,
                            "actors": image_actors,
                        }, f, indent=2)

            done_so_far = batch_start + len(batch)
            logger.info(
                f"  [{done_so_far}/{len(image_paths)}]  "
                f"faces detected={total_faces}  recognised={total_recognised}"
            )

        logger.info(
            f"[ActorTag] Done — {total_recognised}/{total_faces} faces recognised "
            f"across {len(all_results)} images"
        )
        return all_results


def get_tagger_session(
    actor_embeddings_dir: str | Path,
    yolo_model_path: str | Path,
    gpu_id: int = 0,
    similarity_threshold: float = 0.35,
    similarity_margin: float = 0.10,
    cast_filter: list[str] | None = None,
    actor_gender_map: dict[str, str] | None = None,
    yolo_batch_size: int = 32,
    yolo_imgsz: int = 640,
    yolo_conf: float = 0.5,
) -> ActorTaggerSession:
    key = _session_key(
        Path(actor_embeddings_dir),
        Path(yolo_model_path),
        gpu_id,
        similarity_threshold,
        similarity_margin,
        cast_filter,
        actor_gender_map,
        yolo_batch_size,
        yolo_imgsz,
        yolo_conf,
    )
    if key not in _SESSIONS:
        _SESSIONS[key] = ActorTaggerSession(
            Path(actor_embeddings_dir),
            Path(yolo_model_path),
            gpu_id,
            similarity_threshold,
            similarity_margin,
            cast_filter,
            actor_gender_map,
            yolo_batch_size,
            yolo_imgsz,
            yolo_conf,
        )
    return _SESSIONS[key]


def warm_session(
    actor_embeddings_dir: str | Path,
    yolo_model_path: str | Path,
    gpu_id: int = 0,
    similarity_threshold: float = 0.35,
    similarity_margin: float = 0.10,
    cast_filter: list[str] | None = None,
    actor_gender_map: dict[str, str] | None = None,
    yolo_batch_size: int = 32,
    yolo_imgsz: int = 640,
    yolo_conf: float = 0.5,
) -> ActorTaggerSession:
    """Eagerly load models (call once per Ray actor at startup)."""
    return get_tagger_session(
        actor_embeddings_dir,
        yolo_model_path,
        gpu_id,
        similarity_threshold,
        similarity_margin,
        cast_filter,
        actor_gender_map,
        yolo_batch_size,
        yolo_imgsz,
        yolo_conf,
    )


def tag_frames(
    image_paths: list,
    actor_embeddings_dir: str | Path,
    output_dir: str | Path,
    yolo_model_path: str | Path,
    gpu_id: int = 0,
    similarity_threshold: float = 0.35,
    similarity_margin: float = 0.10,
    cast_filter: list[str] | None = None,
    actor_gender_map: dict[str, str] | None = None,
    yolo_batch_size: int = 32,
    yolo_imgsz: int = 640,
    yolo_conf: float = 0.5,
) -> dict[str, list[dict]]:
    """Detect and identify actors in a list of image paths.

    For each image with ≥1 recognised actor (similarity ≥ threshold):
      - Writes output_dir/{parentdir}__{stem}_actors.json

    Returns:
        dict: image_path_str → [{"actor", "display_name", "similarity", "bbox", "yolo_confidence"}]
        Only images with at least one recognised actor are included.
    """
    image_paths = [Path(p) for p in image_paths]
    if not image_paths:
        logger.info("[ActorTag] No images to process.")
        return {}

    try:
        session = get_tagger_session(
            actor_embeddings_dir,
            yolo_model_path,
            gpu_id,
            similarity_threshold,
            similarity_margin,
            cast_filter,
            actor_gender_map,
            yolo_batch_size,
            yolo_imgsz,
            yolo_conf,
        )
    except RuntimeError as e:
        logger.warning(f"[ActorTag] {e}")
        return {}

    return session.tag_frames(image_paths, Path(output_dir))
