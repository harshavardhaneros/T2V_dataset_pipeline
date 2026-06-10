"""Ray GPU actors for s5 classify and s8 video caption."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def _bootstrap() -> None:
    root = os.environ.get("INDIC_PIPELINE_ROOT", "")
    if root and root not in sys.path:
        sys.path.insert(0, root)


_bootstrap()

try:
    import ray
except ImportError:
    ray = None  # type: ignore


if ray is not None:

    @ray.remote(num_gpus=1)
    class QwenVideoCaptionActor:
        """One Qwen2.5-VL replica on a single Ray-assigned GPU."""

        def __init__(self, config: Dict[str, Any]):
            _bootstrap()
            from common.qwen_video_caption import QwenVideoCaptionWorker

            self._worker = QwenVideoCaptionWorker(config, device="cuda:0")

        def caption(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            from common.qwen_video_caption import build_video_caption_prompt
            from common.gemma_caption import parse_caption_json, to_single_line_json
            from common.actor_caption import enforce_actor_names_in_caption
            from common.screen_position import known_actor_names

            rec = payload["record"]
            clip_path = Path(payload["clip_path"])
            try:
                prompt = build_video_caption_prompt(rec, payload["config"])
                raw = self._worker.caption_video(clip_path, prompt)
                actors = rec.get("clip_actors") or known_actor_names(
                    rec.get("actors") or []
                )
                if actors:
                    struct = parse_caption_json(raw)
                    short = struct.get("short_description", "")
                    if short:
                        struct["short_description"] = enforce_actor_names_in_caption(
                            short, actors
                        )
                        raw = to_single_line_json(
                            json.dumps(struct, ensure_ascii=False)
                        )
                return {"clip_id": rec["clip_id"], "raw": raw, "ok": True}
            except Exception as exc:
                return {"clip_id": rec["clip_id"], "raw": "", "ok": False, "error": str(exc)}

        def shutdown(self) -> None:
            self._worker.cleanup()

    @ray.remote(num_gpus=1)
    class QwenClassifyActor:
        """Fast bucket classify (7B recommended) on one GPU."""

        def __init__(self, config: Dict[str, Any]):
            _bootstrap()
            from common.qwen_classify import QwenClassifyWorker

            self._worker = QwenClassifyWorker(config, device="cuda:0")

        def classify(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            return self._worker.classify_clip(payload)

        def shutdown(self) -> None:
            self._worker.cleanup()

    @ray.remote(num_gpus=1)
    class GemmaVerifyActor:
        """Gemma-4B bucket verify — one replica per GPU."""

        def __init__(self, config: Dict[str, Any]):
            _bootstrap()
            from copy import deepcopy

            from common.gemma_verify import GemmaVerifyService

            local = deepcopy(config)
            local.setdefault("models", {}).setdefault("gemma", {})["gpu_ids"] = [0]
            self._gemma = GemmaVerifyService(local)
            self._gemma.load()

        def verify(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            from PIL import Image
            import cv2

            rec = payload["record"]
            clip_path = Path(payload["clip_path"])
            try:
                cap = cv2.VideoCapture(str(clip_path))
                cap.set(cv2.CAP_PROP_POS_MSEC, 2500.0)
                ok, frame = cap.read()
                cap.release()
                if not ok:
                    raise ValueError(f"no frame in {clip_path}")
                image = Image.fromarray(frame[:, :, ::-1])
                data = self._gemma.verify(image, rec.get("bucket", "bucket_01"))
                return {"clip_id": rec["clip_id"], **data, "ok": True}
            except Exception as exc:
                return {
                    "clip_id": rec["clip_id"],
                    "verified": False,
                    "confidence": 0.0,
                    "route": "other",
                    "bucket_matches": False,
                    "ok": False,
                    "error": str(exc),
                }

        def shutdown(self) -> None:
            self._gemma.cleanup()

    @ray.remote(num_gpus=1)
    class ActorTagActor:
        """YOLO + InsightFace actor tagging on one GPU."""

        def __init__(self, config: Dict[str, Any], master_cfg: Dict[str, Any]):
            _bootstrap()
            from common.master_bridge import (
                init_master,
                ensure_yolo_face_model,
                warm_actor_tagger,
            )
            from common.paths import yolo_face_model_path

            init_master(master_cfg["root"])
            ensure_yolo_face_model(yolo_face_model_path(config))
            self.config = config
            self.master_cfg = master_cfg
            # Ray remaps assigned GPU to logical cuda:0 inside each actor.
            warm_actor_tagger({**master_cfg, "actor_tag_gpu_id": 0})

        def _tag_payloads(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            from common.master_bridge import tag_actor_frames

            if not payloads:
                return []

            tags_dir = Path(payloads[0]["tags_dir"])
            local_cfg = {**self.master_cfg, "actor_tag_gpu_id": 0}
            clip_meta: List[tuple[Dict[str, Any], Dict[int, Path]]] = []
            all_paths: List[Path] = []

            for payload in payloads:
                rec = payload["record"]
                frame_map = {
                    int(k): Path(v) for k, v in payload["frame_map"].items() if v
                }
                paths = [
                    frame_map[i] for i in sorted(frame_map) if frame_map[i].exists()
                ]
                clip_meta.append((rec, frame_map))
                all_paths.extend(paths)

            tag_results = tag_actor_frames(all_paths, local_cfg, tags_dir)
            rows: List[Dict[str, Any]] = []
            for rec, frame_map in clip_meta:
                frame_assignments = {
                    idx: tag_results.get(str(fp), []) for idx, fp in frame_map.items()
                }
                rows.append({
                    "clip_id": rec["clip_id"],
                    "frame_assignments": frame_assignments,
                    "frame_map": {k: str(v) for k, v in frame_map.items()},
                })
            return rows

        def tag_clips_batch(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return self._tag_payloads(payloads)

        def tag_clip(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            rows = self._tag_payloads([payload])
            return rows[0] if rows else {}

    @ray.remote(num_gpus=1)
    class ClipScoreActor:
        """OpenCLIP alignment scoring — one replica per GPU."""

        def __init__(self, config: Dict[str, Any]):
            _bootstrap()
            from model_clients.clip_client import ClipClient

            self._config = config
            self._client = ClipClient(config.get("models", config))

        def score_clip(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            from common.caption_text import caption_to_str
            from common.frame_sampler import sample_keyframes

            rec = payload["record"]
            weights = payload["weights"]
            fractions = payload["fractions"]
            clip_path = str(payload["clip_path"])
            clip_len = float(payload["clip_length_sec"])
            caption = caption_to_str(rec.get("caption")) or "Indic cultural scene"

            frames = sample_keyframes(clip_path, 0.0, clip_len, fractions=fractions)
            if frames:
                scores = [self._client.score_image_text(f, caption) for f in frames]
                clip_score = sum(scores) / len(scores)
            else:
                clip_score = 0.0

            verified = rec.get("bucket_verified")
            if verified is None:
                verified = rec.get("verified", False)
            bucket_sem = (
                max(0.0, min(1.0, float(rec.get("bucket_confidence", 0) or 0)))
                if verified
                else 0.0
            )
            dover = max(0.0, min(1.0, float(rec.get("dover_score", 0) or 0)))
            motion = max(0.0, min(1.0, float(rec.get("motion_score", 0) or 0)))
            cap_ok = 1.0 if caption and caption.strip() else 0.0
            final = (
                weights["clip_score"] * clip_score
                + weights["dover_score"] * dover
                + weights["motion_score"] * motion
                + weights["bucket_semantic"] * bucket_sem
                + weights["caption_present"] * cap_ok
            )
            return {
                "clip_id": rec["clip_id"],
                "clip_score": round(clip_score, 4),
                "icr": round(bucket_sem, 4),
                "aod": round(dover, 4),
                "final_score": round(final, 4),
            }

    @ray.remote(num_gpus=1)
    class MotionScoreActor:
        """UniMatch + VMAF motion on one Ray-assigned GPU."""

        def score(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            _bootstrap()
            from copy import deepcopy

            from common.clip_workers import score_clip_motion

            local = deepcopy(payload)
            mc = deepcopy(local.get("model_cfg", {}))
            uni = dict(mc.get("unimatch", {}))
            uni["device"] = "cuda:0"
            mc["unimatch"] = uni
            local["model_cfg"] = mc
            return score_clip_motion(local)

    @ray.remote(num_gpus=1)
    class DoverScoreActor:
        """DOVER quality scoring on one Ray-assigned GPU."""

        def __init__(self, config: Dict[str, Any]):
            _bootstrap()
            from copy import deepcopy

            from model_clients.dover_client import DoverClient

            local = deepcopy(config)
            models = local.setdefault("models", {})
            dover = dict(models.get("dover", {}))
            dover["device"] = "cuda:0"
            models["dover"] = dover
            self._client = DoverClient(local)

        def score(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            record = payload["record"]
            clip_path = str(payload["clip_path"])
            scores = self._client.score_video(clip_path)
            return {
                "clip_id": record["clip_id"],
                "aesthetic_score": round(float(scores["aesthetic_score"]), 4),
                "technical_score": round(float(scores["technical_score"]), 4),
                "dover_score": round(float(scores["dover_score"]), 4),
            }

else:
    QwenVideoCaptionActor = None  # type: ignore
    QwenClassifyActor = None  # type: ignore
    GemmaVerifyActor = None  # type: ignore
    ActorTagActor = None  # type: ignore
    ClipScoreActor = None  # type: ignore
    MotionScoreActor = None  # type: ignore
    DoverScoreActor = None  # type: ignore
