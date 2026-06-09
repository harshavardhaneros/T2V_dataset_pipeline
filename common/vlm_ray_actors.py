"""Ray GPU actors for s5 classify and s8 video caption."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


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

else:
    QwenVideoCaptionActor = None  # type: ignore
    QwenClassifyActor = None  # type: ignore
