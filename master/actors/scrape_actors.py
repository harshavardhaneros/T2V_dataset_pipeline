#!/usr/bin/env python3
"""
Indian Actor Image Scraper + Embedding Builder

Downloads high-quality face images for Indian cinema actors from multiple
search engines, validates face presence via YOLO, and builds InsightFace
embeddings for the actor-tagging pipeline.

Usage:
    # Scrape all actors (skip existing)
    python3 scrape_actors.py

    # Scrape + rebuild embeddings
    python3 scrape_actors.py --build-embeddings

    # Scrape specific actors only
    python3 scrape_actors.py --actors "Alia Bhatt" "Prabhas"

    # Force re-scrape existing actors
    python3 scrape_actors.py --force

    # Just rebuild embeddings from existing images
    python3 scrape_actors.py --embeddings-only
"""

import argparse
import hashlib
import logging
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
ACTOR_IMAGES_DIR = SCRIPT_DIR / "actor_images"
ACTOR_EMBEDDINGS_DIR = SCRIPT_DIR / "actor_embeddings"

# Minimum requirements for a usable face image
MIN_IMAGE_SIZE = 150        # px (width or height)
MIN_FACE_SIZE = 80          # px (face bbox width)
MAX_IMAGES_PER_ACTOR = 60   # download up to this many, keep best after filtering
TARGET_IMAGES_PER_ACTOR = 50  # aim for this many usable images


# ── Comprehensive Indian Actor List ──────────────────────────────────────────
# Format: (display_name, search_queries)
# Multiple queries improve diversity (portrait, movie still, event photo)

INDIAN_ACTORS = [
    # ═══ BOLLYWOOD — Male ═══════════════════════════════════════════════════
    ("Amitabh Bachchan", ["Amitabh Bachchan face portrait", "Amitabh Bachchan actor close up"]),
    ("Shah Rukh Khan", ["Shah Rukh Khan face portrait", "Shah Rukh Khan actor headshot"]),
    ("Aamir Khan", ["Aamir Khan face portrait", "Aamir Khan actor close up"]),
    ("Salman Khan", ["Salman Khan actor face portrait", "Salman Khan Bollywood headshot"]),
    ("Akshay Kumar", ["Akshay Kumar face portrait", "Akshay Kumar actor close up"]),
    ("Hrithik Roshan", ["Hrithik Roshan face portrait", "Hrithik Roshan actor headshot"]),
    ("Ranveer Singh", ["Ranveer Singh face portrait", "Ranveer Singh actor close up"]),
    ("Ranbir Kapoor", ["Ranbir Kapoor face portrait", "Ranbir Kapoor actor headshot"]),
    ("Varun Dhawan", ["Varun Dhawan face portrait", "Varun Dhawan actor close up"]),
    ("Ayushmann Khurrana", ["Ayushmann Khurrana face portrait", "Ayushmann Khurrana actor"]),
    ("Rajkummar Rao", ["Rajkummar Rao face portrait", "Rajkummar Rao actor close up"]),
    ("Vicky Kaushal", ["Vicky Kaushal face portrait", "Vicky Kaushal actor headshot"]),
    ("Kartik Aaryan", ["Kartik Aaryan face portrait", "Kartik Aaryan actor close up"]),
    ("Shahid Kapoor", ["Shahid Kapoor face portrait", "Shahid Kapoor actor headshot"]),
    ("Tiger Shroff", ["Tiger Shroff face portrait", "Tiger Shroff actor close up"]),
    ("Irrfan Khan", ["Irrfan Khan actor face portrait", "Irrfan Khan close up"]),
    ("Nawazuddin Siddiqui", ["Nawazuddin Siddiqui face portrait", "Nawazuddin Siddiqui actor"]),
    ("Pankaj Tripathi", ["Pankaj Tripathi face portrait", "Pankaj Tripathi actor close up"]),
    ("Manoj Bajpayee", ["Manoj Bajpayee face portrait", "Manoj Bajpayee actor headshot"]),
    ("Anil Kapoor", ["Anil Kapoor face portrait", "Anil Kapoor actor close up"]),
    ("Govinda", ["Govinda actor face portrait", "Govinda Bollywood actor close up"]),
    ("Sanjay Dutt", ["Sanjay Dutt face portrait", "Sanjay Dutt actor headshot"]),
    ("Ajay Devgn", ["Ajay Devgn face portrait", "Ajay Devgn actor close up"]),
    ("Sunny Deol", ["Sunny Deol face portrait", "Sunny Deol actor close up"]),
    ("Bobby Deol", ["Bobby Deol face portrait", "Bobby Deol actor close up"]),
    ("Arjun Kapoor", ["Arjun Kapoor face portrait", "Arjun Kapoor actor close up"]),
    ("Sidharth Malhotra", ["Sidharth Malhotra face portrait", "Sidharth Malhotra actor"]),
    ("John Abraham", ["John Abraham actor face portrait", "John Abraham actor close up"]),
    ("Abhishek Bachchan", ["Abhishek Bachchan face portrait", "Abhishek Bachchan actor"]),

    # ═══ BOLLYWOOD — Female ═════════════════════════════════════════════════
    ("Deepika Padukone", ["Deepika Padukone face portrait", "Deepika Padukone actress close up"]),
    ("Priyanka Chopra", ["Priyanka Chopra face portrait", "Priyanka Chopra actress close up"]),
    ("Alia Bhatt", ["Alia Bhatt face portrait", "Alia Bhatt actress close up"]),
    ("Kareena Kapoor", ["Kareena Kapoor face portrait", "Kareena Kapoor actress headshot"]),
    ("Katrina Kaif", ["Katrina Kaif face portrait", "Katrina Kaif actress close up"]),
    ("Anushka Sharma", ["Anushka Sharma face portrait", "Anushka Sharma actress close up"]),
    ("Kangana Ranaut", ["Kangana Ranaut face portrait", "Kangana Ranaut actress headshot"]),
    ("Kriti Sanon", ["Kriti Sanon face portrait", "Kriti Sanon actress close up"]),
    ("Shraddha Kapoor", ["Shraddha Kapoor face portrait", "Shraddha Kapoor actress close up"]),
    ("Janhvi Kapoor", ["Janhvi Kapoor face portrait", "Janhvi Kapoor actress close up"]),
    ("Sara Ali Khan", ["Sara Ali Khan face portrait", "Sara Ali Khan actress close up"]),
    ("Kiara Advani", ["Kiara Advani face portrait", "Kiara Advani actress headshot"]),
    ("Taapsee Pannu", ["Taapsee Pannu face portrait", "Taapsee Pannu actress close up"]),
    ("Vidya Balan", ["Vidya Balan face portrait", "Vidya Balan actress close up"]),
    ("Madhuri Dixit", ["Madhuri Dixit face portrait", "Madhuri Dixit actress close up"]),
    ("Aishwarya Rai", ["Aishwarya Rai face portrait", "Aishwarya Rai actress headshot"]),
    ("Sridevi", ["Sridevi actress face portrait", "Sridevi Indian actress close up"]),
    ("Rekha", ["Rekha actress face portrait", "Rekha Bollywood actress close up"]),
    ("Hema Malini", ["Hema Malini face portrait", "Hema Malini actress close up"]),
    ("Rani Mukerji", ["Rani Mukerji face portrait", "Rani Mukerji actress close up"]),
    ("Preity Zinta", ["Preity Zinta face portrait", "Preity Zinta actress close up"]),
    ("Juhi Chawla", ["Juhi Chawla face portrait", "Juhi Chawla actress close up"]),
    ("Kajol", ["Kajol actress face portrait", "Kajol Bollywood actress close up"]),
    ("Parineeti Chopra", ["Parineeti Chopra face portrait", "Parineeti Chopra actress"]),
    ("Sonakshi Sinha", ["Sonakshi Sinha face portrait", "Sonakshi Sinha actress close up"]),
    ("Disha Patani", ["Disha Patani face portrait", "Disha Patani actress close up"]),
    ("Rashmika Mandanna", ["Rashmika Mandanna face portrait", "Rashmika Mandanna actress"]),

    # ═══ TOLLYWOOD (Telugu) ═════════════════════════════════════════════════
    ("Prabhas", ["Prabhas actor face portrait", "Prabhas Telugu actor close up"]),
    ("Mahesh Babu", ["Mahesh Babu face portrait", "Mahesh Babu actor close up"]),
    ("Allu Arjun", ["Allu Arjun face portrait", "Allu Arjun actor headshot"]),
    ("NTR Jr", ["NTR Jr face portrait", "Jr NTR actor close up"]),
    ("Ram Charan", ["Ram Charan actor face portrait", "Ram Charan Telugu actor close up"]),
    ("Chiranjeevi", ["Chiranjeevi actor face portrait", "Chiranjeevi Telugu actor close up"]),
    ("Nagarjuna", ["Nagarjuna actor face portrait", "Nagarjuna Akkineni close up"]),
    ("Venkatesh", ["Venkatesh Daggubati face portrait", "Venkatesh Telugu actor close up"]),
    ("Nani", ["Nani Telugu actor face portrait", "Nani actor close up"]),
    ("Ravi Teja", ["Ravi Teja actor face portrait", "Ravi Teja Telugu actor close up"]),
    ("Samantha Ruth Prabhu", ["Samantha Ruth Prabhu face portrait", "Samantha actress close up"]),
    ("Pooja Hegde", ["Pooja Hegde face portrait", "Pooja Hegde actress close up"]),
    ("Anushka Shetty", ["Anushka Shetty face portrait", "Anushka Shetty actress close up"]),
    ("Kajal Aggarwal", ["Kajal Aggarwal face portrait", "Kajal Aggarwal actress close up"]),

    # ═══ KOLLYWOOD (Tamil) ══════════════════════════════════════════════════
    ("Rajinikanth", ["Rajinikanth face portrait", "Rajinikanth actor close up"]),
    ("Kamal Haasan", ["Kamal Haasan face portrait", "Kamal Haasan actor close up"]),
    ("Vijay", ["Thalapathy Vijay face portrait", "Vijay Tamil actor close up"]),
    ("Ajith Kumar", ["Ajith Kumar face portrait", "Ajith Kumar actor close up"]),
    ("Suriya", ["Suriya actor face portrait", "Suriya Tamil actor close up"]),
    ("Dhanush", ["Dhanush actor face portrait", "Dhanush Tamil actor close up"]),
    ("Vikram", ["Vikram actor face portrait", "Chiyaan Vikram actor close up"]),
    ("Karthi", ["Karthi actor face portrait", "Karthi Tamil actor close up"]),
    ("Sivakarthikeyan", ["Sivakarthikeyan face portrait", "Sivakarthikeyan actor close up"]),
    ("Nayanthara", ["Nayanthara face portrait", "Nayanthara actress close up"]),
    ("Trisha Krishnan", ["Trisha Krishnan face portrait", "Trisha actress close up"]),
    ("Jyothika", ["Jyothika actress face portrait", "Jyothika Tamil actress close up"]),
    ("Ramya Krishnan", ["Ramya Krishnan face portrait", "Ramya Krishnan actress close up"]),
    ("Namitha", ["Namitha actress face portrait", "Namitha Tamil actress close up"]),
    ("Shriya Saran", ["Shriya Saran face portrait", "Shriya Saran actress close up"]),

    # ═══ KANNADA ════════════════════════════════════════════════════════════
    ("Yash", ["Yash Kannada actor face portrait", "Yash KGF actor close up"]),
    ("Darshan", ["Darshan Thoogudeepa face portrait", "Darshan Kannada actor close up"]),
    ("Sudeep", ["Kiccha Sudeep face portrait", "Sudeep Kannada actor close up"]),
    ("Puneeth Rajkumar", ["Puneeth Rajkumar face portrait", "Puneeth Rajkumar actor"]),
    ("Upendra", ["Upendra Kannada actor face portrait", "Upendra actor close up"]),

    # ═══ MALAYALAM ══════════════════════════════════════════════════════════
    ("Mohanlal", ["Mohanlal face portrait", "Mohanlal actor close up"]),
    ("Mammootty", ["Mammootty face portrait", "Mammootty actor close up"]),
    ("Dulquer Salmaan", ["Dulquer Salmaan face portrait", "Dulquer Salmaan actor close up"]),
    ("Fahadh Faasil", ["Fahadh Faasil face portrait", "Fahadh Faasil actor close up"]),
    ("Prithviraj Sukumaran", ["Prithviraj Sukumaran face portrait", "Prithviraj actor"]),
    ("Tovino Thomas", ["Tovino Thomas face portrait", "Tovino Thomas actor close up"]),
    ("Manju Warrier", ["Manju Warrier face portrait", "Manju Warrier actress close up"]),

    # ═══ CHARACTER ACTORS / PAN-INDIA ═══════════════════════════════════════
    ("Ashish Vidyarthi", ["Ashish Vidyarthi actor face portrait", "Ashish Vidyarthi close up"]),
    ("Sayaji Shinde", ["Sayaji Shinde actor face portrait", "Sayaji Shinde close up"]),
    ("Prakash Raj", ["Prakash Raj actor face portrait", "Prakash Raj close up"]),
    ("Naseeruddin Shah", ["Naseeruddin Shah face portrait", "Naseeruddin Shah actor close up"]),
    ("Om Puri", ["Om Puri actor face portrait", "Om Puri close up"]),
    ("Anupam Kher", ["Anupam Kher face portrait", "Anupam Kher actor close up"]),
    ("Boman Irani", ["Boman Irani face portrait", "Boman Irani actor close up"]),
    ("Paresh Rawal", ["Paresh Rawal face portrait", "Paresh Rawal actor close up"]),
    ("Rajpal Yadav", ["Rajpal Yadav face portrait", "Rajpal Yadav actor close up"]),
    ("Vijay Sethupathi", ["Vijay Sethupathi face portrait", "Vijay Sethupathi actor close up"]),
    ("Manoj Bajpayee", ["Manoj Bajpayee face portrait", "Manoj Bajpayee actor headshot"]),
]


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:120]


def _download_image(url: str, timeout: int = 15) -> bytes | None:
    """Download image bytes with retries."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type and not url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            return None
        data = resp.content
        if len(data) < 5000:  # too small, probably a placeholder
            return None
        return data
    except Exception:
        return None


def _validate_image(img_bytes: bytes) -> Image.Image | None:
    """Validate image is real, minimum size, and loadable."""
    try:
        from io import BytesIO
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
            return None
        return img
    except Exception:
        return None


def scrape_actor_images_bing(
    actor_name: str,
    queries: list[str],
    output_dir: Path,
    max_images: int = MAX_IMAGES_PER_ACTOR,
) -> int:
    """Scrape images using Bing Image Search via icrawler."""
    from icrawler.builtin import BingImageCrawler

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(output_dir.glob("*.jpg")))
    if existing >= TARGET_IMAGES_PER_ACTOR:
        logger.info(f"  {actor_name}: already has {existing} images, skipping scrape")
        return existing

    remaining = max_images - existing
    per_query = max(remaining // len(queries), 15)

    temp_dir = output_dir / "_temp_scrape"
    temp_dir.mkdir(exist_ok=True)

    for query in queries:
        try:
            crawler = BingImageCrawler(
                storage={"root_dir": str(temp_dir)},
                log_level=logging.WARNING,
            )
            crawler.crawl(
                keyword=query,
                max_num=per_query,
                min_size=(200, 200),
                filters={"type": "photo", "size": "medium"},
            )
        except Exception as e:
            logger.warning(f"  Bing crawl failed for '{query}': {e}")

    # Move and deduplicate downloaded images
    downloaded = 0
    seen_hashes = set()

    # Load existing image hashes to avoid dupes
    for existing_img in output_dir.glob("*.jpg"):
        try:
            h = hashlib.md5(existing_img.read_bytes()).hexdigest()[:16]
            seen_hashes.add(h)
        except Exception:
            pass

    for img_path in sorted(temp_dir.glob("*")):
        if img_path.is_dir():
            continue
        try:
            data = img_path.read_bytes()
            h = hashlib.md5(data).hexdigest()[:16]
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            img = _validate_image(data)
            if img is None:
                continue

            # Save as standardized JPEG
            fname = hashlib.md5(data).hexdigest()[:10] + ".jpg"
            dest = output_dir / fname
            if not dest.exists():
                img.save(dest, "JPEG", quality=92)
                downloaded += 1
        except Exception:
            pass

    # Cleanup temp
    shutil.rmtree(temp_dir, ignore_errors=True)
    total = len(list(output_dir.glob("*.jpg")))
    return total


def scrape_actor_images_google(
    actor_name: str,
    queries: list[str],
    output_dir: Path,
    max_images: int = MAX_IMAGES_PER_ACTOR,
) -> int:
    """Scrape images using Google Image Search via icrawler."""
    from icrawler.builtin import GoogleImageCrawler

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(output_dir.glob("*.jpg")))
    if existing >= TARGET_IMAGES_PER_ACTOR:
        return existing

    remaining = max_images - existing
    per_query = max(remaining // len(queries), 15)

    temp_dir = output_dir / "_temp_scrape"
    temp_dir.mkdir(exist_ok=True)

    for query in queries:
        try:
            crawler = GoogleImageCrawler(
                storage={"root_dir": str(temp_dir)},
                log_level=logging.WARNING,
            )
            crawler.crawl(
                keyword=query,
                max_num=per_query,
                min_size=(200, 200),
            )
        except Exception as e:
            logger.warning(f"  Google crawl failed for '{query}': {e}")

    # Process downloaded images
    downloaded = 0
    seen_hashes = set()
    for existing_img in output_dir.glob("*.jpg"):
        try:
            h = hashlib.md5(existing_img.read_bytes()).hexdigest()[:16]
            seen_hashes.add(h)
        except Exception:
            pass

    for img_path in sorted(temp_dir.glob("*")):
        if img_path.is_dir():
            continue
        try:
            data = img_path.read_bytes()
            h = hashlib.md5(data).hexdigest()[:16]
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            img = _validate_image(data)
            if img is None:
                continue

            fname = hashlib.md5(data).hexdigest()[:10] + ".jpg"
            dest = output_dir / fname
            if not dest.exists():
                img.save(dest, "JPEG", quality=92)
                downloaded += 1
        except Exception:
            pass

    shutil.rmtree(temp_dir, ignore_errors=True)
    return len(list(output_dir.glob("*.jpg")))


def filter_images_with_yolo(
    actor_dir: Path,
    yolo_model_path: str | Path,
    min_face_size: int = MIN_FACE_SIZE,
) -> int:
    """Remove images where YOLO can't detect a face or face is too small.

    Returns number of images remaining after filtering.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning("YOLO not available, skipping face validation")
        return len(list(actor_dir.glob("*.jpg")))

    images = sorted(actor_dir.glob("*.jpg"))
    if not images:
        return 0

    yolo = YOLO(str(yolo_model_path))
    # Use CPU for this validation to avoid GPU contention
    removed = 0

    for img_path in images:
        try:
            results = yolo.predict(
                source=str(img_path),
                verbose=False,
                device="cpu",
                imgsz=640,
                conf=0.4,
            )
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                img_path.unlink()
                removed += 1
                continue

            # Check face size
            boxes = results[0].boxes.xyxy.cpu().numpy()
            max_face_w = max(b[2] - b[0] for b in boxes)
            if max_face_w < min_face_size:
                img_path.unlink()
                removed += 1
        except Exception:
            pass

    remaining = len(list(actor_dir.glob("*.jpg")))
    if removed > 0:
        logger.info(f"    Removed {removed} images without valid faces, {remaining} remaining")
    return remaining


def build_embeddings(
    actor_images_dir: Path,
    embeddings_dir: Path,
    gpu_id: int = 0,
    force: bool = False,
):
    """Build InsightFace embeddings for all actor directories."""
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    from actor_tagger import build_actor_embeddings

    build_actor_embeddings(
        actor_images_dir=str(actor_images_dir),
        output_dir=str(embeddings_dir),
        gpu_id=gpu_id,
        force=force,
    )


def scrape_all(
    actors: list[tuple[str, list[str]]] | None = None,
    force: bool = False,
    build_emb: bool = False,
    embeddings_only: bool = False,
    yolo_filter: bool = True,
    gpu_id: int = 0,
    engine: str = "bing",
):
    """Main entry point: scrape images for all actors and optionally build embeddings."""
    if actors is None:
        actors = INDIAN_ACTORS

    ACTOR_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ACTOR_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    yolo_path = SCRIPT_DIR / "yolov12n-face.pt"

    if not embeddings_only:
        logger.info(f"Scraping images for {len(actors)} actors using {engine}")
        logger.info(f"Output: {ACTOR_IMAGES_DIR}")

        for i, (name, queries) in enumerate(actors, 1):
            slug = slugify(name)
            actor_dir = ACTOR_IMAGES_DIR / slug
            existing = len(list(actor_dir.glob("*.jpg"))) if actor_dir.exists() else 0

            if existing >= TARGET_IMAGES_PER_ACTOR and not force:
                logger.info(f"[{i}/{len(actors)}] {name} — {existing} images (skip)")
                continue

            logger.info(f"[{i}/{len(actors)}] {name} — scraping ({existing} existing)...")

            # Try primary engine
            if engine == "google":
                count = scrape_actor_images_google(name, queries, actor_dir)
            else:
                count = scrape_actor_images_bing(name, queries, actor_dir)

            # If primary engine didn't get enough, try the other
            if count < TARGET_IMAGES_PER_ACTOR // 2:
                logger.info(f"    Only {count} images, trying alternate engine...")
                if engine == "google":
                    count = scrape_actor_images_bing(name, queries, actor_dir)
                else:
                    count = scrape_actor_images_google(name, queries, actor_dir)

            logger.info(f"    {name}: {count} images downloaded")

            # YOLO face validation
            if yolo_filter and yolo_path.exists() and count > 0:
                filter_images_with_yolo(actor_dir, yolo_path)

            # Rate limit between actors
            time.sleep(2)

    if build_emb or embeddings_only:
        logger.info("Building InsightFace embeddings...")
        build_embeddings(
            ACTOR_IMAGES_DIR,
            ACTOR_EMBEDDINGS_DIR,
            gpu_id=gpu_id,
            force=force,
        )

    # Print summary
    total_actors = 0
    total_images = 0
    total_embeddings = 0
    for d in sorted(ACTOR_IMAGES_DIR.iterdir()):
        if d.is_dir():
            n = len(list(d.glob("*.jpg")))
            if n > 0:
                total_actors += 1
                total_images += n
    total_embeddings = len(list(ACTOR_EMBEDDINGS_DIR.glob("*.pkl")))

    logger.info("=" * 60)
    logger.info(f"  SUMMARY")
    logger.info(f"  Actors with images:  {total_actors}")
    logger.info(f"  Total images:        {total_images}")
    logger.info(f"  Embeddings built:    {total_embeddings}")
    logger.info(f"  Images dir:          {ACTOR_IMAGES_DIR}")
    logger.info(f"  Embeddings dir:      {ACTOR_EMBEDDINGS_DIR}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Indian Actor Image Scraper")
    parser.add_argument("--actors", nargs="+", help="Specific actor names to scrape")
    parser.add_argument("--force", action="store_true", help="Re-scrape existing actors")
    parser.add_argument("--build-embeddings", action="store_true", help="Build embeddings after scraping")
    parser.add_argument("--embeddings-only", action="store_true", help="Only rebuild embeddings")
    parser.add_argument("--no-yolo-filter", action="store_true", help="Skip YOLO face validation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for embedding generation")
    parser.add_argument("--engine", choices=["bing", "google"], default="bing", help="Search engine")
    parser.add_argument("--list-actors", action="store_true", help="Print actor list and exit")
    args = parser.parse_args()

    if args.list_actors:
        print(f"\n{len(INDIAN_ACTORS)} actors in database:\n")
        for i, (name, _) in enumerate(INDIAN_ACTORS, 1):
            slug = slugify(name)
            actor_dir = ACTOR_IMAGES_DIR / slug
            n = len(list(actor_dir.glob("*.jpg"))) if actor_dir.exists() else 0
            emb = ACTOR_EMBEDDINGS_DIR / f"{slug}.pkl"
            status = f"{n} imgs" + (" + emb" if emb.exists() else "")
            print(f"  {i:3d}. {name:<30s}  [{status}]")
        return

    # Filter to specific actors if requested
    actor_list = None
    if args.actors:
        requested = {slugify(a) for a in args.actors}
        actor_list = [(n, q) for n, q in INDIAN_ACTORS if slugify(n) in requested]
        if not actor_list:
            # Try partial match
            actor_list = [
                (n, q) for n, q in INDIAN_ACTORS
                if any(req in slugify(n) for req in requested)
            ]
        if not actor_list:
            logger.error(f"No matching actors found for: {args.actors}")
            logger.info("Use --list-actors to see available actors")
            return

    scrape_all(
        actors=actor_list,
        force=args.force,
        build_emb=args.build_embeddings,
        embeddings_only=args.embeddings_only,
        yolo_filter=not args.no_yolo_filter,
        gpu_id=args.gpu,
        engine=args.engine,
    )


if __name__ == "__main__":
    main()
