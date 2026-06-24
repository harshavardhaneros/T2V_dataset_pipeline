#!/usr/bin/env python3
"""
Actor Image Validator — removes wrong-person images via face clustering.

Problem: Web-scraped images often include spouses, co-stars, group shots.
Solution: Extract face embeddings from ALL images per actor, find the dominant
cluster (= the actual actor), and remove images that don't match.

Algorithm per actor:
  1. YOLO face detection → crop largest face per image
  2. InsightFace embedding (512-d) per face
  3. Cosine similarity matrix → find the "consensus face"
     (the embedding most similar to all others = the real actor)
  4. Remove images below similarity threshold to the consensus
  5. Rebuild averaged embedding from cleaned set

Usage:
    # Validate all actors (dry-run first to see what would be removed)
    python3 validate_actors.py --dry-run

    # Actually remove bad images + rebuild embeddings
    python3 validate_actors.py --clean --rebuild

    # Validate specific actor
    python3 validate_actors.py --actors suriya --dry-run

    # Adjust strictness (default 0.35, higher = stricter)
    python3 validate_actors.py --clean --threshold 0.40
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
ACTOR_IMAGES_DIR = SCRIPT_DIR / "actor_images"
ACTOR_EMBEDDINGS_DIR = SCRIPT_DIR / "actor_embeddings"
QUARANTINE_DIR = SCRIPT_DIR / "actor_images_quarantine"

# If an image's best face has similarity < this to the consensus, it's removed
DEFAULT_THRESHOLD = 0.35

# Need at least this many images to form a reliable consensus
MIN_IMAGES_FOR_CONSENSUS = 5

# Images with multiple faces where actor face is small relative to others → suspect
MULTI_FACE_MIN_RATIO = 0.3  # actor face must be at least 30% of largest face area


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:120]


def _init_models(gpu_id: int = 0):
    """Initialize YOLO + InsightFace once."""
    from ultralytics import YOLO
    from insightface.app import FaceAnalysis
    import torch

    yolo_path = SCRIPT_DIR / "yolov12n-face.pt"
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {yolo_path}")

    use_gpu = torch.cuda.is_available() and gpu_id >= 0
    device = f"cuda:{gpu_id}" if use_gpu else "cpu"

    yolo = YOLO(str(yolo_path))
    yolo.to(device)

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_gpu else ["CPUExecutionProvider"]
    )
    face_app = FaceAnalysis(name="buffalo_l", providers=providers)
    face_app.prepare(ctx_id=gpu_id if use_gpu else -1, det_size=(640, 640))

    return yolo, face_app, device


def _extract_face_embedding(
    img_path: Path,
    face_app,
) -> tuple[np.ndarray | None, int, float]:
    """Extract the largest face embedding from an image.

    Returns:
        (embedding_or_None, num_faces_detected, face_area_ratio)
        face_area_ratio: area of largest face / area of second largest (if multi-face)
    """
    try:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            return None, 0, 0.0

        faces = face_app.get(bgr)
        if not faces:
            return None, 0, 0.0

        # Sort by face area (largest first)
        faces_sorted = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )

        best = faces_sorted[0]
        best_area = (best.bbox[2] - best.bbox[0]) * (best.bbox[3] - best.bbox[1])

        # Check face size relative to image
        img_h, img_w = bgr.shape[:2]
        face_ratio = best_area / (img_h * img_w + 1e-9)

        # For multi-face images, check if the biggest face dominates
        area_ratio = 1.0
        if len(faces_sorted) > 1:
            second_area = (faces_sorted[1].bbox[2] - faces_sorted[1].bbox[0]) * \
                          (faces_sorted[1].bbox[3] - faces_sorted[1].bbox[1])
            area_ratio = best_area / (second_area + 1e-9)

        emb = best.normed_embedding.astype(np.float32)
        return emb, len(faces), area_ratio

    except Exception as e:
        logger.debug(f"  Error processing {img_path.name}: {e}")
        return None, 0, 0.0


def validate_actor(
    actor_dir: Path,
    face_app,
    threshold: float = DEFAULT_THRESHOLD,
    dry_run: bool = True,
) -> dict:
    """Validate images for a single actor using consensus face clustering.

    Returns dict with validation stats.
    """
    actor_name = actor_dir.name
    images = sorted(actor_dir.glob("*.jpg"))

    if len(images) < MIN_IMAGES_FOR_CONSENSUS:
        logger.info(f"  {actor_name}: only {len(images)} images, skipping validation")
        return {"actor": actor_name, "total": len(images), "kept": len(images),
                "removed": 0, "no_face": 0, "skipped": True}

    # Step 1: Extract embeddings for all images
    embeddings = {}  # path → embedding
    no_face = []
    multi_face_suspect = []

    for img_path in images:
        emb, n_faces, area_ratio = _extract_face_embedding(img_path, face_app)
        if emb is None:
            no_face.append(img_path)
        else:
            embeddings[img_path] = emb
            # Flag group photos where actor face might not be the biggest
            if n_faces >= 3:
                multi_face_suspect.append(img_path)

    if len(embeddings) < MIN_IMAGES_FOR_CONSENSUS:
        logger.info(f"  {actor_name}: only {len(embeddings)} faces detected, skipping")
        return {"actor": actor_name, "total": len(images), "kept": len(images),
                "removed": 0, "no_face": len(no_face), "skipped": True}

    # Step 2: Build similarity matrix → find consensus face
    paths = list(embeddings.keys())
    emb_matrix = np.stack([embeddings[p] for p in paths])  # [N, 512]

    # Normalize
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix = emb_matrix / (norms + 1e-8)

    # Cosine similarity matrix [N, N]
    sim_matrix = emb_matrix @ emb_matrix.T

    # Consensus = the embedding with highest average similarity to all others
    # This person appears most consistently → the real actor
    avg_sims = sim_matrix.mean(axis=1)
    consensus_idx = int(np.argmax(avg_sims))
    consensus_emb = emb_matrix[consensus_idx]
    consensus_path = paths[consensus_idx]

    logger.info(f"  {actor_name}: consensus face from {consensus_path.name} "
                f"(avg_sim={avg_sims[consensus_idx]:.3f})")

    # Step 3: Score each image against consensus
    scores = (emb_matrix @ consensus_emb.reshape(-1, 1)).flatten()

    to_remove = []
    to_keep = []

    for i, (path, score) in enumerate(zip(paths, scores)):
        if score < threshold:
            to_remove.append((path, float(score)))
        else:
            to_keep.append((path, float(score)))

    # Also remove no-face images
    to_remove_nf = [(p, 0.0) for p in no_face]

    # Sort removals by score for nice logging
    to_remove.sort(key=lambda x: x[1])

    total_remove = len(to_remove) + len(to_remove_nf)
    total_keep = len(to_keep)

    # Log results
    if to_remove or to_remove_nf:
        logger.info(f"  {actor_name}: KEEP {total_keep}, REMOVE {total_remove} "
                     f"({len(to_remove)} wrong person, {len(to_remove_nf)} no face)")
        for path, score in to_remove[:5]:
            logger.info(f"    REMOVE: {path.name} (sim={score:.3f})")
        if len(to_remove) > 5:
            logger.info(f"    ... and {len(to_remove) - 5} more")
    else:
        logger.info(f"  {actor_name}: all {total_keep} images valid")

    # Step 4: Actually remove (or quarantine) bad images
    if not dry_run:
        quarantine_actor = QUARANTINE_DIR / actor_name
        quarantine_actor.mkdir(parents=True, exist_ok=True)

        for path, score in to_remove + to_remove_nf:
            dest = quarantine_actor / path.name
            shutil.move(str(path), str(dest))

    return {
        "actor": actor_name,
        "total": len(images),
        "kept": total_keep,
        "removed": total_remove,
        "no_face": len(to_remove_nf),
        "wrong_person": len(to_remove),
        "consensus_img": consensus_path.name,
        "consensus_avg_sim": float(avg_sims[consensus_idx]),
        "skipped": False,
        "removed_files": [p.name for p, _ in to_remove + to_remove_nf] if not dry_run else [],
    }


def validate_all(
    threshold: float = DEFAULT_THRESHOLD,
    dry_run: bool = True,
    rebuild: bool = False,
    actors: list[str] | None = None,
    gpu_id: int = 0,
):
    """Validate all actor directories."""
    logger.info("Initializing face detection models...")
    yolo, face_app, device = _init_models(gpu_id)

    # Collect actor dirs
    actor_dirs = sorted(
        d for d in ACTOR_IMAGES_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )

    if actors:
        slugs = {slugify(a) for a in actors}
        actor_dirs = [d for d in actor_dirs if d.name in slugs or
                      any(s in d.name for s in slugs)]

    logger.info(f"Validating {len(actor_dirs)} actors (threshold={threshold}, "
                f"dry_run={dry_run})")

    if not dry_run:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Quarantine dir: {QUARANTINE_DIR}")

    results = []
    total_removed = 0
    total_kept = 0

    for i, actor_dir in enumerate(actor_dirs, 1):
        logger.info(f"[{i}/{len(actor_dirs)}] {actor_dir.name}")
        r = validate_actor(actor_dir, face_app, threshold=threshold, dry_run=dry_run)
        results.append(r)
        total_removed += r["removed"]
        total_kept += r["kept"]

    # Summary
    logger.info("=" * 60)
    logger.info(f"  VALIDATION {'DRY RUN' if dry_run else 'COMPLETE'}")
    logger.info(f"  Actors validated:  {len(results)}")
    logger.info(f"  Total kept:        {total_kept}")
    logger.info(f"  Total removed:     {total_removed}")
    logger.info(f"  Threshold:         {threshold}")
    if dry_run:
        logger.info(f"  (Run with --clean to actually remove)")
    logger.info("=" * 60)

    # Show worst offenders
    by_removed = sorted(results, key=lambda r: -r["removed"])
    if by_removed and by_removed[0]["removed"] > 0:
        logger.info("\n  Actors with most removals:")
        for r in by_removed[:15]:
            if r["removed"] == 0:
                break
            logger.info(f"    {r['actor']:<30s}  remove {r['removed']:>3d}/{r['total']}"
                        f"  (no_face={r['no_face']}, wrong={r.get('wrong_person', 0)})")

    # Save report
    report_path = SCRIPT_DIR / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump({"threshold": threshold, "dry_run": dry_run, "results": results}, f, indent=2)
    logger.info(f"\n  Report: {report_path}")

    # Rebuild embeddings if requested
    if rebuild and not dry_run:
        logger.info("\nRebuilding embeddings from cleaned images...")
        sys.path.insert(0, str(SCRIPT_DIR.parent))
        from actor_tagger import build_actor_embeddings
        build_actor_embeddings(
            actor_images_dir=str(ACTOR_IMAGES_DIR),
            output_dir=str(ACTOR_EMBEDDINGS_DIR),
            gpu_id=gpu_id,
            force=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Actor Image Validator")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be removed without actually removing")
    parser.add_argument("--clean", action="store_true",
                        help="Actually move bad images to quarantine")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild embeddings after cleaning")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Similarity threshold (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--actors", nargs="+",
                        help="Specific actor names/slugs to validate")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID for face models")
    args = parser.parse_args()

    if not args.clean and not args.dry_run:
        args.dry_run = True
        logger.info("No --clean flag, defaulting to --dry-run")

    validate_all(
        threshold=args.threshold,
        dry_run=not args.clean,
        rebuild=args.rebuild,
        actors=args.actors,
        gpu_id=args.gpu,
    )


if __name__ == "__main__":
    main()
