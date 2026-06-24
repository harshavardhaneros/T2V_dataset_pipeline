#!/usr/bin/env python3
"""
Structured captioning with 12-bucket prompts for Indian cultural images.

Loads bucket-specific prompt templates from T2I_bucket_prompts/, then uses
Qwen2.5-VL-32B to produce structured JSON captions (caption + tags).

Input: directory of classified images with sidecar JSON from classifier.py
Output: sidecar *_caption.json files with structured caption + tags

Usage:
    python captioner.py --input-dir /path/to/classified_frames \
        --prompt-dir /data/kl_dev/prompts/T2I_bucket_prompts --gpus 0,1

    python captioner.py --input-dir /path/to/frames \
        --bucket people_portraits --gpus 0
"""

import argparse
import gc
import json
import time
import logging
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from common import IMAGE_EXTS, MODEL_ID, unique_stem, parse_llm_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Map bucket short names to prompt file names
BUCKET_PROMPT_FILES = {
    "people_portraits":    "bucket_01_people_portraits.txt",
    "clothing_textiles":   "bucket_02_clothing_textiles.txt",
    "architecture":        "bucket_03_architecture_built_environment.txt",
    "landscape_nature":    "bucket_04_landscape_nature.txt",
    "urban_street":        "bucket_05_urban_street_life.txt",
    "rural_village":       "bucket_06_rural_village_life.txt",
    "food_drink":          "bucket_07_food_drink.txt",
    "festivals_rituals":   "bucket_08_festivals_rituals_events.txt",
    "objects_artifacts":   "bucket_09_objects_artifacts.txt",
    "animals_wildlife":    "bucket_10_animals_wildlife.txt",
    "art_design":          "bucket_11_art_design_creative.txt",
    "abstract_texture":    "bucket_12_abstract_texture_pattern.txt",
}

# Aliases to normalize category names from classifier.py
BUCKET_ALIASES = {
    "people_portraits": "people_portraits",
    "clothing_textiles": "clothing_textiles",
    "clothing_costume": "clothing_textiles",
    "architecture": "architecture",
    "architecture_built_environment": "architecture",
    "architecture_location": "architecture",
    "landscape_nature": "landscape_nature",
    "urban_street": "urban_street",
    "urban_street_life": "urban_street",
    "rural_village": "rural_village",
    "rural_village_life": "rural_village",
    "food_drink": "food_drink",
    "food_market": "food_drink",
    "festivals_rituals": "festivals_rituals",
    "festivals_rituals_events": "festivals_rituals",
    "festival_ritual": "festivals_rituals",
    "objects_artifacts": "objects_artifacts",
    "animals_wildlife": "animals_wildlife",
    "art_design": "art_design",
    "art_design_creative": "art_design",
    "abstract_texture": "abstract_texture",
    "abstract_texture_pattern": "abstract_texture",
    "everyday_life": "urban_street",
}


JSON_SCHEMA_REMINDER = """

OUTPUT FORMAT (return only this JSON, nothing else):
{
  "caption": "• <Sentence 1: subjects/objects + actions> • <Sentence 2: location/setting> • <Sentence 3: aesthetics — lighting, colour, mood, visual style> • <Sentence 4: camera angle, framing, focal point>",
  "tags": {
    "setting":       "<indoor | outdoor | studio | mixed>",
    "lighting":      "<natural daylight | golden hour | warm artificial | cool artificial | overcast | night | high contrast | soft diffused>",
    "composition":   "<close-up | mid-shot | full body | overhead flat lay | 45-degree | wide establishing | aerial | detail crop | dutch angle>",
    "mood":          "<e.g. warm, inviting | dramatic, tense | festive, vibrant | serene, contemplative>",
    "color_palette": "<warm earth tones | cool blues | vibrant saturated | monochrome | pastel | high contrast B&W>",
    "image_angle":   "<eye-level | low angle | high angle | bird's eye | worm's eye | tilted>",
    "subject_focus": "<single subject | multiple subjects | crowd | object | landscape | abstract>",
    "time_of_day":   "<morning | afternoon | evening | night | golden hour | unknown>",
    "era_style":     "<contemporary | vintage | black and white film | 1980s-90s | historical | timeless>"
  }
}
"""


def load_prompts(prompt_dir: Path) -> dict[str, str]:
    """Load all bucket prompt templates from text files.

    If a prompt file does not already contain the JSON OUTPUT FORMAT schema,
    appends it automatically so Qwen returns structured JSON.
    """
    prompts = {}
    for bucket, filename in BUCKET_PROMPT_FILES.items():
        path = prompt_dir / filename
        if path.exists():
            text = path.read_text().strip()
            # Append JSON schema if not already embedded in the prompt
            if '"tags"' not in text or '"caption"' not in text:
                text += JSON_SCHEMA_REMINDER
            prompts[bucket] = text
        else:
            logger.warning(f"Prompt file not found: {path}")
    logger.info(f"Loaded {len(prompts)} bucket prompts")
    return prompts


def normalize_bucket(category: str) -> str:
    """Normalize various category names to canonical bucket names."""
    return BUCKET_ALIASES.get(category, "people_portraits")


def load_model(gpu_ids: list[int], model_path: str | None = None,
               backend: str = "transformers"):
    """Load VLM model. Returns a VLMBackend instance."""
    from vlm_backend import create_backend
    b = create_backend(backend, model_path=model_path or MODEL_ID,
                       gpu_ids=gpu_ids, max_new_tokens=512)
    b.load()
    return b


def parse_caption_json(raw: str) -> dict:
    """Parse VLM output into structured caption dict."""
    fallback = {"caption": raw.strip()[:500], "tags": {}, "_parse_error": True}
    return parse_llm_json(raw, fallback=fallback)


def collect_images(
    input_dir: Path,
    bucket_filter: str | None = None,
    actor_tags_dir: Path | None = None,
) -> list[dict]:
    """Collect images with their classification metadata.

    Looks for sidecar .json files from classifier.py, or processes
    all images if no sidecar exists. Propagates source_type from vlm_results.
    If actor_tags_dir is provided, loads actor tag JSONs and adds "actors" field.
    """
    items = []
    input_dir = input_dir.resolve()

    # Check for VLM result JSONs in a vlm_results/ subdirectory
    vlm_dir = input_dir / "vlm_results"
    if vlm_dir.exists():
        for jp in sorted(vlm_dir.glob("*.json")):
            with open(jp) as f:
                data = json.load(f)
            if data.get("rejected", True):
                continue
            img_path = Path(data.get("image", ""))
            if not img_path.exists():
                continue
            category = normalize_bucket(data.get("category", ""))
            if bucket_filter and category != bucket_filter:
                continue
            item = {
                "image_path": img_path,
                "category": category,
                "description": data.get("description", ""),
                "source_type": data.get("source_type", ""),
            }
            # For precaptioned images, read co-located .txt as original caption
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                item["original_caption"] = txt_path.read_text(
                    encoding="utf-8", errors="replace"
                ).strip()
            _attach_actors(item, actor_tags_dir)
            items.append(item)
    else:
        # Scan directory directly for images
        for img_path in sorted(input_dir.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            # Check for sidecar JSON
            sidecar = img_path.with_suffix(".json")
            category = "people_portraits"  # default
            description = ""
            source_type = ""
            if sidecar.exists():
                with open(sidecar) as f:
                    data = json.load(f)
                if data.get("rejected", True):
                    continue
                category = normalize_bucket(data.get("category", ""))
                description = data.get("description", "")
                source_type = data.get("source_type", "")

            if bucket_filter and category != bucket_filter:
                continue
            item = {
                "image_path": img_path,
                "category": category,
                "description": description,
                "source_type": source_type,
            }
            _attach_actors(item, actor_tags_dir)
            items.append(item)

    return items


def _attach_actors(item: dict, actor_tags_dir: Path | None) -> None:
    """Load actor tags sidecar and attach to item in-place (if available)."""
    if actor_tags_dir is None:
        return
    img_path = item["image_path"]
    ustem = _unique_stem(img_path)
    actor_json = actor_tags_dir / f"{ustem}_actors.json"
    if actor_json.exists():
        try:
            with open(actor_json) as f:
                at = json.load(f)
            actors = at.get("actors", [])
            if actors:
                item["actors"] = actors
        except Exception:
            pass


def _unique_stem(img_path: Path) -> str:
    """Alias for _common.unique_stem — kept for backward compat with CLI."""
    return unique_stem(img_path)


def run_captioning(
    input_dir: Path,
    prompt_dir: Path,
    output_dir: Path,
    gpu_ids: list[int],
    bucket_filter: str | None = None,
    max_new_tokens: int = 512,
    skip_existing: bool = True,
    model_path: str | None = None,
    backend: str = "transformers",
    actor_tags_dir: Path | None = None,
):
    """Run structured captioning on classified images.

    If actor_tags_dir is set, actor names detected by step_tag_actors are
    injected into the prompt so the model generates named captions
    (e.g. "Shah Rukh Khan in a white kurta…" instead of "A man in a white kurta…").
    """
    prompts = load_prompts(prompt_dir)
    if not prompts:
        logger.error("No prompts loaded. Check --prompt-dir.")
        return

    items = collect_images(input_dir, bucket_filter, actor_tags_dir=actor_tags_dir)
    if not items:
        logger.error("No images found to caption.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Captioning {len(items)} images -> {output_dir}")

    # Filter already captioned (using collision-safe naming)
    if skip_existing:
        filtered = []
        for item in items:
            out_path = output_dir / f"{_unique_stem(item['image_path'])}_caption.json"
            if out_path.exists():
                continue
            filtered.append(item)
        skipped = len(items) - len(filtered)
        if skipped:
            logger.info(f"Skipping {skipped} already captioned images")
        items = filtered

    if not items:
        logger.info("All images already captioned!")
        return

    vlm_backend = load_model(gpu_ids, model_path=model_path, backend=backend)

    stats = {"total": 0, "success": 0, "parse_error": 0, "error": 0}
    bucket_counts = {}

    # For vLLM, use batch captioning for major speedup
    try:
        from vlm_backend import VLLMBackend
        use_batch = isinstance(vlm_backend, VLLMBackend)
    except ImportError:
        use_batch = False

    if use_batch:
        _run_batch_captioning(vlm_backend, items, prompts, output_dir, max_new_tokens, stats, bucket_counts)
    else:
        _run_sequential_captioning(vlm_backend, items, prompts, output_dir, max_new_tokens, stats, bucket_counts)

    # Cleanup
    vlm_backend.cleanup()

    # Summary
    logger.info(f"Captioning complete:")
    logger.info(f"  Total:       {stats['total']}")
    logger.info(f"  Success:     {stats['success']}")
    logger.info(f"  Parse Error: {stats['parse_error']}")
    logger.info(f"  Error:       {stats['error']}")
    logger.info(f"  Per-bucket counts:")
    for b, c in sorted(bucket_counts.items()):
        logger.info(f"    {b:30s}  {c}")


def _build_prompt(item: dict, prompts: dict) -> str:
    """Build the final captioning prompt, injecting actor names when available."""
    category = item["category"]
    prompt = prompts.get(category, prompts.get("people_portraits", "Describe this image."))

    # Inject actor names — overrides generic references so the VLM uses real names
    actors = item.get("actors", [])
    if actors:
        names = ", ".join(a["display_name"] for a in actors)
        prompt = (
            f"IMPORTANT: The person(s) in this image are: {names}. "
            f"Always refer to them by their exact name(s). "
            f"Never use generic words like 'a man', 'a woman', or 'a person'.\n\n"
        ) + prompt

    if item.get("original_caption"):
        label = item["original_caption"].strip()
        # Short labels (< 100 chars) are cultural names/terms — inject as hard fact
        # Long text (> 100 chars) is a full description — skip it, let VLM caption fresh
        if len(label) <= 100 and label:
            prompt = (
                f"IMPORTANT: This image contains: {label}. "
                f"You MUST use this exact name/term in your caption. "
                f"Do not replace it with generic words.\n\n"
            ) + prompt

    return prompt


def _run_sequential_captioning(vlm_backend, items, prompts, output_dir, max_new_tokens, stats, bucket_counts):
    """Sequential captioning (transformers backend)."""
    for i, item in enumerate(tqdm(items, desc="Captioning")):
        img_path = item["image_path"]
        category = item["category"]
        prompt = _build_prompt(item, prompts)

        ustem = _unique_stem(img_path)
        out_path = output_dir / f"{ustem}_caption.json"
        stats["total"] += 1

        try:
            pil = Image.open(img_path).convert("RGB")
            max_side = 1024
            w, h = pil.size
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            t0 = time.time()
            raw = vlm_backend.generate(pil, prompt)
            elapsed = round(time.time() - t0, 2)

            parsed = parse_caption_json(raw)

            result = {
                "image": str(img_path),
                "image_name": img_path.name,
                "bucket": category,
                "source_type": item.get("source_type", ""),
                "caption": parsed.get("caption", ""),
                "tags": parsed.get("tags", {}),
                "model": MODEL_ID,
                "inference_time_s": elapsed,
            }
            if parsed.get("_parse_error"):
                result["_parse_error"] = True
                stats["parse_error"] += 1
            else:
                stats["success"] += 1

            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            txt_path = output_dir / f"{ustem}_recaptioned.txt"
            with open(txt_path, "w") as f:
                f.write(parsed.get("caption", ""))

            bucket_counts[category] = bucket_counts.get(category, 0) + 1

        except Exception as e:
            logger.error(f"Error captioning {img_path.name}: {e}")
            stats["error"] += 1
        finally:
            torch.cuda.empty_cache()

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{len(items)} | "
                        f"Success: {stats['success']} | Errors: {stats['error']}")


def _run_batch_captioning(vlm_backend, items, prompts, output_dir, max_new_tokens, stats, bucket_counts):
    """Batched captioning (vLLM backend) for 10-15x throughput."""
    BATCH_SIZE = 16
    for batch_start in range(0, len(items), BATCH_SIZE):
        batch = items[batch_start:batch_start + BATCH_SIZE]
        batch_inputs = []
        batch_meta = []

        for item in batch:
            img_path = item["image_path"]
            category = item["category"]
            prompt = _build_prompt(item, prompts)

            try:
                pil = Image.open(img_path).convert("RGB")
                max_side = 1024
                w, h = pil.size
                if max(w, h) > max_side:
                    scale = max_side / max(w, h)
                    pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                batch_inputs.append((pil, prompt))
                batch_meta.append(item)
            except Exception as e:
                logger.error(f"Error loading {img_path.name}: {e}")
                stats["error"] += 1

        if not batch_inputs:
            continue

        stats["total"] += len(batch_inputs)
        t0 = time.time()
        try:
            raw_outputs = vlm_backend.generate_batch(batch_inputs)
        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            stats["error"] += len(batch_inputs)
            continue
        elapsed = round(time.time() - t0, 2)

        for raw, item in zip(raw_outputs, batch_meta):
            img_path = item["image_path"]
            category = item["category"]
            ustem = _unique_stem(img_path)
            out_path = output_dir / f"{ustem}_caption.json"

            parsed = parse_caption_json(raw)
            result = {
                "image": str(img_path),
                "image_name": img_path.name,
                "bucket": category,
                "source_type": item.get("source_type", ""),
                "caption": parsed.get("caption", ""),
                "tags": parsed.get("tags", {}),
                "model": MODEL_ID,
                "inference_time_s": round(elapsed / len(batch_inputs), 2),
            }
            if parsed.get("_parse_error"):
                result["_parse_error"] = True
                stats["parse_error"] += 1
            else:
                stats["success"] += 1

            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            txt_path = output_dir / f"{ustem}_recaptioned.txt"
            with open(txt_path, "w") as f:
                f.write(parsed.get("caption", ""))

            bucket_counts[category] = bucket_counts.get(category, 0) + 1

        logger.info(f"Batch [{batch_start+1}-{batch_start+len(batch_inputs)}/{len(items)}] "
                    f"done in {elapsed}s | Success: {stats['success']} | Errors: {stats['error']}")


def main():
    parser = argparse.ArgumentParser(description="Structured captioning with 12-bucket prompts")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory with classified images (from classifier.py)")
    parser.add_argument("--prompt-dir", type=str,
                        default=str(Path(__file__).resolve().parent / "prompts"),
                        help="Directory with bucket prompt .txt files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save caption JSONs")
    parser.add_argument("--gpus", type=str, default="0,1",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--bucket", type=str, default=None,
                        help="Only caption images in this bucket")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-caption existing files")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Local model path (default: uses MODEL_ID from HuggingFace)")
    parser.add_argument("--backend", type=str, default="transformers",
                        choices=["transformers", "vllm"],
                        help="VLM inference backend (default: transformers)")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    bucket_filter = normalize_bucket(args.bucket) if args.bucket else None

    run_captioning(
        input_dir=Path(args.input_dir),
        prompt_dir=Path(args.prompt_dir),
        output_dir=Path(args.output_dir),
        gpu_ids=gpu_ids,
        bucket_filter=bucket_filter,
        max_new_tokens=args.max_new_tokens,
        skip_existing=not args.no_skip,
        model_path=args.model_path,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
