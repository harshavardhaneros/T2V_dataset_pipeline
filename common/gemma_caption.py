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

from common.gpu_info import log_service_gpus, resolve_gpu_ids
from common.paths import models_root
from common.screen_position import frame_position_label, known_actor_names

logger = logging.getLogger(__name__)

# Matches eros_caption_video/pipeline.py CAPTION_SYSTEM_PROMPT + clip-level motion rules.
CAPTION_SYSTEM_PROMPT = (
    "Output MUST be a valid JSON object only. No markdown or extra text.\n\n"
    "Rules:\n"
    "- Be precise and avoid repetition.\n"
    "- No hallucination. Only visible or strongly implied details.\n"
    "- Avoid generic phrases (e.g., \"a group of people\").\n"
    "- For humans, describe from THEIR perspective (not the viewer's).\n"
    "- Prioritise culturally significant visual elements when present.\n"
    "- Include actor names and positions while explaining object actions.\n"
    "- You are captioning a short VIDEO CLIP (about 3 seconds), not a single photograph.\n"
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


def gemma_caption_model_path(config: Dict[str, Any]) -> Path:
    cc = config.get("models", {}).get("gemma_caption", {})
    pcfg = config.get("pipeline", {}).get("captioner", {})
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


def build_caption_user_text(rec: Dict[str, Any], *, multi_frame: bool = False) -> str:
    """User prompt aligned with eros_caption_video/pipeline.py H200Captioner._build_messages."""
    clip_actors = rec.get("clip_actors") or known_actor_names(rec.get("actors") or [])
    lines = [
        f"Actors present: {clip_actors}",
        f"Frame 1: {rec.get('actors_f1', '[]')} | {rec.get('pos_f1', 'unknown')}",
        f"Frame 2: {rec.get('actors_f2', '[]')} | {rec.get('pos_f2', 'unknown')}",
    ]
    if multi_frame:
        lines.insert(
            0,
            "The images are three sequential frames from one 3-second video clip "
            "(at 0.5s, 1.5s, and 2.5s). Describe the full clip, including movement.",
        )
    lines.append(
        "You are a Visual Art Director generating structured, "
        "high-quality captions for the video frames."
    )
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
            raise FileNotFoundError(
                f"Gemma caption model not found: {self.model_path}\n"
                "Download: hf download google/gemma-3-4b-it --local-dir "
                f"{self.model_path}"
            )
        log_service_gpus(
            "s8",
            "Gemma-3-4B-IT caption (eros-style JSON)",
            self.model_path,
            self.gpu_ids,
        )
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()

    def _build_messages(
        self,
        rec: Dict[str, Any],
        frame_paths: list[Path],
    ) -> tuple[list | None, list[Image.Image]]:
        if not frame_paths:
            return None, []
        labels = ("0.5s", "1.5s", "2.5s")
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
                label = labels[i] if i < len(labels) else f"frame {i + 1}"
                content.append({"type": "text", "text": f"Frame at {label} into the clip:"})
            content.append({"type": "image", "image": img})
        if not images:
            return None, []
        content.append({
            "type": "text",
            "text": build_caption_user_text(rec, multi_frame=multi),
        })
        messages = [
            {"role": "system", "content": [{"type": "text", "text": CAPTION_SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        return messages, images

    def _infer_single(self, messages: list) -> str:
        import torch

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=torch.bfloat16)
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
                torch.cuda.empty_cache()

        producer.join()
        gc.collect()
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
    rec["frame1"] = str(frame_paths.get(1, ""))
    rec["frame2"] = str(frame_paths.get(2, ""))
    rec["frame3"] = str(frame_paths.get(3, ""))
    if frame_assignments.get(2):
        rec["actors"] = frame_assignments[2]
    elif frame_assignments.get(1):
        rec["actors"] = frame_assignments[1]
    else:
        rec["actors"] = []
