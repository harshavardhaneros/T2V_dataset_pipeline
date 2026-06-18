"""Gemma-3 vision captioner (eros_caption_video architecture)."""

from __future__ import annotations

import gc
import json
import logging
import queue
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from common.clip_io import frame_offsets_for_record
from common.bucket_prompts import bucket_prompt_for_record
from common.gpu_info import log_service_gpus, resolve_gpu_ids
from common.caption_models import caption_model_key, resolve_caption_model
from common.paths import models_root
from common.screen_position import frame_position_label, known_actor_names

logger = logging.getLogger(__name__)

# Matches eros_caption_video/pipeline.py CAPTION_SYSTEM_PROMPT + clip-level motion rules.
CAPTION_JSON_SYSTEM_PROMPT = (
    "Output MUST be a valid JSON object only. No markdown or extra text.\n\n"
    "Rules:\n"
    "- Be precise and avoid repetition.\n"
    "- No hallucination. Only visible or strongly implied details.\n"
    "- Avoid generic phrases (e.g., \"a group of people\").\n"
    "- For humans, describe from THEIR perspective (not the viewer's).\n"
    "- Prioritise culturally significant visual elements when present.\n"
    "- Include actor names and positions while explaining object actions.\n"
    "- You are captioning a short VIDEO CLIP (not a single photograph).\n"
    "- Use all provided sequential frames to infer motion, actions, and camera movement.\n"
    "- short_description and actor_name_and_action must describe what happens over the clip.\n\n"
    "Indian Cultural Details (include ONLY if visible):\n"
    "- attire: women: saree (silk/cotton), half-saree, salwar, blouse color/design,\n"
    "  embroidery (Zardozi, Chikankari). men: veshti/dhoti, kurta, shirt, traditional wear\n"
    "- accessories: jhumka, nose ring, choker, chain, bangles, anklets, kundan, bindi/sindoor\n"
    "- regional_identity: Tamil, Punjabi, Bengali, etc. (ONLY if clearly inferable)\n"
    "- cultural_context: temple, wedding, ritual, festival, street market, rural/urban India\n"
    "- architecture_landmarks: gopuram, heritage buildings (if visible)\n"
    "- food_elements: traditional dishes (if present)\n\n"
    "Text: Include ONLY clearly visible text. If none → return [].\n\n"
    "JSON structure:\n"
    "{ \"short_description\": \"\",\n"
    "  \"objects\": [{ \"description\":\"\",\"location\":\"\",\"relative_size\":\"\","
    "\"shape_color\":\"\",\"texture\":\"\",\"appearance_details\":\"\","
    "\"relationship\":\"\",\"orientation\":\"\",\"Indian_cultural_details\":{},"
    "\"pose\":\"\",\"expression\":\"\",\"clothing\":\"\","
    "\"actor_name_and_action\":\"\",\"gender\":\"\",\"skin_tone_texture\":\"\" }],\n"
    "  \"background_setting\":\"\",\n"
    "  \"lighting\":{\"conditions\":\"\",\"direction\":\"\",\"shadows\":\"\"},\n"
    "  \"aesthetics\":{\"composition\":\"\",\"color_scheme\":\"\",\"mood_atmosphere\":\"\"},\n"
    "  \"photographic_characteristics\":{\"depth_of_field\":\"\",\"focus\":\"\","
    "\"camera_angle\":\"\",\"camera_movement\":\"\",\"lens_focal_length\":\"\"},\n"
    "  \"style_medium\":\"\",\n"
    "  \"text_render\":[{\"text\":\"\",\"location\":\"\",\"size\":\"\","
    "\"color\":\"\",\"font\":\"\",\"appearance_details\":\"\"}] }"
)

# Plain-text variant: same eros rules without JSON structure.
CAPTION_PROSE_SYSTEM_PROMPT = (
    "Write one rich paragraph describing this video clip. Plain text only — "
    "no JSON, no markdown, no bullet lists, no headings.\n\n"
    "Rules:\n"
    "- Be precise and avoid repetition.\n"
    "- No hallucination. Only visible or strongly implied details.\n"
    "- Avoid generic phrases (e.g., \"a group of people\").\n"
    "- For humans, describe from THEIR perspective (not the viewer's).\n"
    "- Prioritise culturally significant visual elements when present.\n"
    "- Include actor names and positions while explaining object actions.\n"
    "- You are captioning a short VIDEO CLIP (not a single photograph).\n"
    "- Use all provided sequential frames to infer motion, actions, and camera movement.\n"
    "- Describe what happens over the full clip, including movement.\n"
    "- Start directly with the subject or action. Never open with meta phrases like "
    "\"In the video clip\", \"In this clip\", \"The video shows\", \"This video depicts\", "
    "\"The clip shows\", or \"The image shows\".\n\n"
    "Indian Cultural Details (include ONLY if visible):\n"
    "- attire: women: saree (silk/cotton), half-saree, salwar, blouse color/design,\n"
    "  embroidery (Zardozi, Chikankari). men: veshti/dhoti, kurta, shirt, traditional wear\n"
    "- accessories: jhumka, nose ring, choker, chain, bangles, anklets, kundan, bindi/sindoor\n"
    "- regional_identity: Tamil, Punjabi, Bengali, etc. (ONLY if clearly inferable)\n"
    "- cultural_context: temple, wedding, ritual, festival, street market, rural/urban India\n"
    "- architecture_landmarks: gopuram, heritage buildings (if visible)\n"
    "- food_elements: traditional dishes (if present)\n\n"
    "Text: mention only clearly visible on-screen text.\n\n"
    "Cover setting, lighting, mood, composition, camera angle/movement, and visible text "
    "naturally within the paragraph."
)

# Backward-compatible alias (JSON is the default structured format).
CAPTION_SYSTEM_PROMPT = CAPTION_JSON_SYSTEM_PROMPT


def caption_format(config: Dict[str, Any]) -> str:
    """Per-run caption output format: ``prose`` (default) or ``json``."""
    raw = str(
        config.get("pipeline", {}).get("captioner", {}).get("caption_format", "prose")
    ).strip().lower()
    if raw in {"prose", "plain", "text", "normal"}:
        return "prose"
    return "json"


def get_caption_system_prompt(config: Dict[str, Any]) -> str:
    if caption_format(config) == "prose":
        return CAPTION_PROSE_SYSTEM_PROMPT
    return CAPTION_JSON_SYSTEM_PROMPT


def _strip_caption_boilerplate(text: str) -> str:
    """Remove common model meta openers from prose captions."""
    cleaned = text.strip()
    patterns = (
        r"^In the video clip,?\s*",
        r"^In this (video )?clip,?\s*",
        r"^The video clip (shows|depicts|features)\s*",
        r"^This video clip (shows|depicts|features)\s*",
        r"^The clip (shows|depicts|features)\s*",
        r"^This clip (shows|depicts|features)\s*",
        r"^The video (shows|depicts|features)\s*",
        r"^This video (shows|depicts|features)\s*",
    )
    for pat in patterns:
        new = re.sub(pat, "", cleaned, count=1, flags=re.IGNORECASE)
        if new != cleaned:
            cleaned = new.strip()
            break
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()
        else:
            cleaned = "\n".join(lines[1:]).strip()
    return cleaned


def normalize_caption_output(
    raw: str,
    rec: Dict[str, Any],
    config: Dict[str, Any],
) -> tuple[str, str, Dict[str, Any]]:
    """Parse model output into (caption, generated_caption, caption_struct)."""
    from common.actor_caption import enforce_actor_names_for_record

    raw = (raw or "").strip()
    if not raw:
        return "", "", {}

    if caption_format(config) == "prose":
        text = _strip_markdown_fences(raw)
        text = _strip_caption_boilerplate(text)
        text = enforce_actor_names_for_record(text, rec, config)
        return text, text, {"short_description": text, "_format": "prose"}

    gen_line = to_single_line_json(raw)
    struct = parse_caption_json(raw)
    short = struct.get("short_description", "")
    if short:
        struct["short_description"] = enforce_actor_names_for_record(
            short, rec, config
        )
        gen_line = json.dumps(struct, ensure_ascii=False)
    caption = struct.get("short_description") or gen_line
    struct.setdefault("_format", "json")
    return caption, gen_line, struct


def gemma_caption_model_path(config: Dict[str, Any]) -> Path:
    from common.caption_models import resolve_caption_model

    pcfg = config.get("pipeline", {}).get("captioner", {})
    if pcfg.get("caption_model") or pcfg.get("model_path"):
        resolved = resolve_caption_model(config)
        if resolved["family"] == "gemma":
            return resolved["model_path"]

    cc = config.get("models", {}).get("gemma_caption", {})
    path = cc.get("model_path") or pcfg.get("model_path")
    if path:
        return Path(path)
    return models_root(config) / "gemma-3-4b-it"


def to_single_line_json(text: str) -> str:
    cleaned = text.strip()
    if "{" in cleaned and "}" in cleaned:
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        try:
            obj = json.loads(cleaned[start_idx : end_idx + 1])
            return json.dumps(obj, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        obj = json.loads(cleaned)
        return json.dumps(obj, ensure_ascii=False)
    except json.JSONDecodeError:
        return re.sub(r"\s+", " ", cleaned)


def parse_caption_json(text: str) -> Dict[str, Any]:
    line = to_single_line_json(text)
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"short_description": line, "_parse_error": True}


def _clip_duration_sec(rec: Dict[str, Any]) -> float:
    try:
        d = float(rec.get("duration") or 0)
    except (TypeError, ValueError):
        d = 0.0
    return d if d > 0 else 5.0


def build_caption_user_text(
    rec: Dict[str, Any],
    *,
    multi_frame: bool = False,
    frame_offsets: Optional[List[float]] = None,
    bucket_guidance: str = "",
) -> str:
    """User prompt for multi-frame video clip captioning."""
    from common.actor_caption import caption_eligible_actors

    clip_actors = caption_eligible_actors(rec) or []
    lines: List[str] = []
    if bucket_guidance.strip():
        lines.append(bucket_guidance.strip())
    if multi_frame:
        dur = _clip_duration_sec(rec)
        offsets = frame_offsets or [dur * 0.2, dur * 0.5, dur * 0.8]
        times = ", ".join(f"{t:.1f}s" for t in offsets[:3])
        lines.append(
            f"Sequential frames span {dur:g} seconds (at {times}). "
            "Describe the full clip including movement and camera work."
        )
    if len(clip_actors) >= 2:
        cast = " and ".join(clip_actors)
        lines.append(
            f"Identified cast in this clip: {cast}. "
            "You must use each person's full name when describing them. "
            "Never write 'the man', 'the woman', 'the other person', or 'two individuals'."
        )
    elif len(clip_actors) == 1:
        lines.append(
            f"Identified person: {clip_actors[0]}. "
            "Use this full name once when describing them. "
            "If other people are visible, call them 'another woman', 'another man', "
            f"or 'another person' — never reuse {clip_actors[0]}'s name for anyone else. "
            "Do not write 'the man', 'the woman', or 'a person' for the identified person."
        )
    else:
        lines.append("Describe the people, setting, and action visible across the frames.")
    return "\n".join(lines)


def pick_caption_frames(rec: Dict[str, Any], frames_dir: Path) -> list[Path]:
    """All 3 clip frames for temporal captioning (eros extracts 3; we feed all 3 to Gemma)."""
    clip_id = rec["clip_id"]
    paths = [frames_dir / f"{clip_id}.{idx}.jpg" for idx in (1, 2, 3)]
    paths = [p for p in paths if p.exists()]
    if paths:
        return paths
    for idx in (2, 1, 3):
        p = frames_dir / f"{clip_id}.{idx}.jpg"
        if p.exists():
            return [p]
    legacy = frames_dir / f"{clip_id}.jpg"
    return [legacy] if legacy.exists() else []


def pick_caption_frame(rec: Dict[str, Any], frames_dir: Path) -> Optional[Path]:
    frames = pick_caption_frames(rec, frames_dir)
    return frames[0] if frames else None


class GemmaCaptionService:
    """Gemma-3-4B-IT captioner with eros-style batching."""

    _shared: Optional["GemmaCaptionService"] = None

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        cc = config.get("models", {}).get("gemma_caption", {})
        pcfg = config.get("pipeline", {}).get("captioner", {})
        self.model_path = str(gemma_caption_model_path(config))
        self.gpu_ids = resolve_gpu_ids(
            [int(g) for g in cc.get("gpu_ids", pcfg.get("gpu_ids", [0]))]
        )
        self.gpu_id = self.gpu_ids[0] if self.gpu_ids else 0
        self.device = f"cuda:{self.gpu_id}"
        self.batch_size = int(cc.get("batch_size", pcfg.get("batch_size", 8)))
        self.max_new_tokens = int(cc.get("max_tokens", pcfg.get("max_tokens", 1000)))
        self._model = None
        self._processor = None

    @classmethod
    def acquire(cls, config: Dict[str, Any]) -> "GemmaCaptionService":
        if cls._shared is None:
            cls._shared = cls(config)
        return cls._shared

    @classmethod
    def release(cls) -> None:
        if cls._shared:
            cls._shared.cleanup()
        cls._shared = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not Path(self.model_path).joinpath("config.json").exists():
            resolved = resolve_caption_model(self._config)
            raise FileNotFoundError(
                f"Gemma caption model not found: {self.model_path}\n"
                f"Download: hf download {resolved['hf_repo']} --local-dir "
                f"{self.model_path}"
            )
        fmt = caption_format(self._config)
        model_key = caption_model_key(self._config)
        resolved = resolve_caption_model(self._config)
        log_service_gpus(
            "s8",
            f"{resolved['label']} caption (eros-style {fmt})",
            self.model_path,
            self.gpu_ids,
        )
        import torch
        from transformers import AutoProcessor

        from common.attn_backend import resolve_attn_implementation

        self._processor = AutoProcessor.from_pretrained(self.model_path)
        load_kwargs: dict = {
            "dtype": torch.bfloat16,
            "attn_implementation": resolve_attn_implementation(),
        }
        if model_key == "gemma4":
            load_kwargs["device_map"] = "auto"
            try:
                from transformers import Gemma4ForConditionalGeneration

                model_cls = Gemma4ForConditionalGeneration
            except ImportError:
                from transformers import AutoModelForImageTextToText

                model_cls = AutoModelForImageTextToText
                load_kwargs["device_map"] = self.device
        else:
            from transformers import AutoModelForImageTextToText

            model_cls = AutoModelForImageTextToText
            load_kwargs["device_map"] = self.device

        self._model = model_cls.from_pretrained(
            self.model_path,
            **load_kwargs,
        ).eval()

    def _build_messages(
        self,
        rec: Dict[str, Any],
        frame_paths: list[Path],
    ) -> tuple[list | None, list[Image.Image]]:
        if not frame_paths:
            return None, []
        offsets = frame_offsets_for_record(rec, self._config)
        dur = _clip_duration_sec(rec)
        images: list[Image.Image] = []
        content: list[dict] = []
        multi = len(frame_paths) > 1
        for i, frame_path in enumerate(frame_paths):
            if not frame_path.exists():
                continue
            try:
                img = Image.open(frame_path).convert("RGB")
            except Exception as exc:
                logger.warning("Cannot open %s: %s", frame_path, exc)
                continue
            images.append(img)
            if multi:
                t = offsets[i] if i < len(offsets) else dur * (i + 1) / (len(frame_paths) + 1)
                content.append({
                    "type": "text",
                    "text": f"Frame at {t:.1f}s into the clip:",
                })
            content.append({"type": "image", "image": img})
        if not images:
            return None, []
        content.append({
            "type": "text",
            "text": build_caption_user_text(
                rec,
                multi_frame=multi,
                frame_offsets=offsets,
                bucket_guidance=bucket_prompt_for_record(
                    rec, self._config, prompt_mgr=self._config.get("_prompt_manager")
                ),
            ),
        })
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": get_caption_system_prompt(self._config)}
                ],
            },
            {"role": "user", "content": content},
        ]
        return messages, images

    def _model_device(self):
        import torch

        if caption_model_key(self._config) == "gemma4":
            return next(self._model.parameters()).device
        return torch.device(self.device)

    def _infer_single(self, messages: list) -> str:
        import torch

        device = self._model_device()
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            gen_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        return self._processor.decode(
            gen_ids[0][input_len:], skip_special_tokens=True
        ).strip()

    def caption_records(
        self,
        items: List[tuple[Dict[str, Any], list[Path]]],
    ) -> List[str]:
        """Caption (metadata_record, frame_paths) pairs. Single-item inference (eros-style)."""
        import torch

        self.load()
        if not items:
            return []

        results = [""] * len(items)
        q: queue.Queue = queue.Queue(maxsize=16)
        SENTINEL = object()

        def _producer():
            for i, (rec, fps) in enumerate(items):
                msgs, imgs = self._build_messages(rec, fps)
                q.put((i, msgs, imgs))
            q.put(SENTINEL)

        producer = threading.Thread(target=_producer, daemon=True)
        producer.start()

        while True:
            item = q.get()
            if item is SENTINEL:
                break
            i, msgs, imgs = item
            if msgs is None or not imgs:
                continue
            try:
                results[i] = self._infer_single(msgs)
            except Exception as exc:
                logger.warning("Caption failed for item %s: %s", i, exc)
            finally:
                for img in imgs:
                    try:
                        img.close()
                    except Exception:
                        pass

        producer.join()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    def cleanup(self) -> None:
        import torch

        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def enrich_record_actor_fields(
    rec: Dict[str, Any],
    frame_assignments: Dict[int, List[Dict[str, Any]]],
    frame_paths: Dict[int, Path],
) -> None:
    """Populate eros-style actor fields on metadata record."""
    all_names: List[str] = []
    for idx in (1, 2, 3):
        actors = frame_assignments.get(idx, [])
        names = known_actor_names(actors)
        rec[f"actors_f{idx}"] = names
        hw = None
        if actors and actors[0].get("_img_hw"):
            hw = actors[0]["_img_hw"]
        elif frame_paths.get(idx) and frame_paths[idx].exists():
            import cv2
            img = cv2.imread(str(frame_paths[idx]))
            if img is not None:
                hw = (img.shape[0], img.shape[1])
        rec[f"pos_f{idx}"] = frame_position_label(actors, hw)
        for n in names:
            if n not in all_names:
                all_names.append(n)
    rec["clip_actors"] = all_names

    sims: List[float] = []
    margins: List[float] = []
    for idx in (1, 2, 3):
        for actor in frame_assignments.get(idx, []):
            if "similarity" in actor:
                sims.append(float(actor["similarity"]))
            if "similarity_margin" in actor:
                margins.append(float(actor["similarity_margin"]))
    rec["actor_tag_min_similarity"] = min(sims) if sims else 0.0
    rec["actor_tag_min_margin"] = min(margins) if margins else 0.0

    rec["frame1"] = str(frame_paths.get(1, ""))
    rec["frame2"] = str(frame_paths.get(2, ""))
    rec["frame3"] = str(frame_paths.get(3, ""))
    from common.actor_caption import best_faces_for_caption

    frame_actors = (
        frame_assignments.get(2)
        or frame_assignments.get(1)
        or frame_assignments.get(3)
        or []
    )
    rec["actors"] = best_faces_for_caption(frame_actors)
