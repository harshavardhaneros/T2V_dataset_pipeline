#!/usr/bin/env python3
"""VCInspector inference worker — batched vLLM video (runs in the vcinspector env).

VCInspector-7B is a merged Qwen2.5-VL-7B model; vLLM serves it natively with the
video modality. Clips are fed as native video (sampled frames + metadata) and
verified against their captions, batched in chunks for high throughput.

CLI is unchanged so services/service_13_caption_verify drives it as before.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_VIDEO_MAX_PIXELS = 50176  # 224x224, per VCInspector README (fast + intended)

PROMPT_SUFFIX = (
    "You are given a video and a caption describing the video content. "
    "Please rate the helpfulness, relevance, accuracy, level of details of the caption. "
    "The overall score should be on a scale of 1 to 5, where a higher score indicates "
    "better overall performance. Please first output a single line containing only one "
    "integer indicating the score. In the subsequent line, please provide a comprehensive "
    "explanation of your evaluation, avoiding any potential bias. STRICTLY FOLLOW THE FORMAT."
)


def build_prompt(caption: str) -> str:
    return f"<caption>{caption}</caption>\n\n{PROMPT_SUFFIX}"


def parse_response(text: str) -> Tuple[Optional[int], str]:
    text = (text or "").strip()
    if not text:
        return None, ""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^\s*(\d)\s*$", line.strip())
        if m:
            score = int(m.group(1))
            if 1 <= score <= 5:
                return score, "\n".join(lines[i + 1 :]).strip()
        m2 = re.match(r"^\s*(\d)\b", line.strip())
        if m2:
            score = int(m2.group(1))
            if 1 <= score <= 5:
                return score, "\n".join(lines[i + 1 :]).strip() or line.strip()
    return None, text


def chunked(items: List[Any], size: int) -> List[List[Any]]:
    size = max(1, size)
    return [items[i : i + size] for i in range(0, len(items), size)]


def run_vllm_inference(
    manifest: List[Dict[str, str]],
    *,
    model: str,
    batch_size: int,
    num_frames: int,
    video_max_pixels: int,
    max_tokens: int,
) -> Dict[str, Dict[str, Any]]:
    os.environ.setdefault("VIDEO_MAX_PIXELS", str(video_max_pixels))
    os.environ.setdefault("FPS_MAX_FRAMES", str(num_frames))

    from concurrent.futures import ThreadPoolExecutor

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.multimodal.video import OpenCVVideoBackend

    processor = AutoProcessor.from_pretrained(model)
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=32768,
        limit_mm_per_prompt={"video": 1},
        enforce_eager=True,
        gpu_memory_utilization=float(
            os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.85")
        ),
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    results: Dict[str, Dict[str, Any]] = {}

    # Video decoding (cv2) is the s13 bottleneck and releases the GIL, so a thread
    # pool parallelizes it. We also PREFETCH the next chunk's frames while the GPU
    # generates the current chunk — decode no longer serializes with inference.
    decode_workers = max(
        1, min(batch_size, int(os.environ.get("S13_DECODE_WORKERS", "12")))
    )

    import math

    import cv2
    import numpy as np

    def _downscale(frames):
        """Shrink frames to <= video_max_pixels area each so vLLM's video
        processor (the 's13 rendering' bottleneck) has far less pixel work."""
        if frames is None or len(frames) == 0:
            return frames
        h, w = frames.shape[1], frames.shape[2]
        if h * w <= video_max_pixels:
            return frames
        scale = math.sqrt(video_max_pixels / float(h * w))
        nw, nh = max(2, int(w * scale)), max(2, int(h * scale))
        out = np.empty((frames.shape[0], nh, nw, frames.shape[3]), dtype=frames.dtype)
        for i in range(frames.shape[0]):
            out[i] = cv2.resize(frames[i], (nw, nh), interpolation=cv2.INTER_AREA)
        return out

    def _decode_one(item: Dict[str, str]):
        clip_path = Path(item["clip_path"])
        if not clip_path.exists():
            return item, None, None, f"clip not found: {clip_path}"
        try:
            with open(clip_path, "rb") as fh:
                frames, metadata = OpenCVVideoBackend.load_bytes(
                    fh.read(), num_frames=num_frames
                )
            frames = _downscale(frames)
            return item, frames, metadata, None
        except Exception as exc:  # noqa: BLE001
            return item, None, None, str(exc)

    chunks = chunked(manifest, batch_size)

    with ThreadPoolExecutor(max_workers=decode_workers) as pool:
        def submit_chunk(idx: int):
            if 0 <= idx < len(chunks):
                return [pool.submit(_decode_one, item) for item in chunks[idx]]
            return []

        pending = submit_chunk(0)
        for idx in range(len(chunks)):
            futures = pending
            pending = submit_chunk(idx + 1)  # prefetch next chunk's decode now

            prompts: List[Dict[str, Any]] = []
            valid: List[Dict[str, str]] = []
            for fut in futures:
                item, frames, metadata, err = fut.result()
                if err:
                    results[item["clip_id"]] = {"score": None, "error": err}
                    continue
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video"},
                            {"type": "text", "text": build_prompt(item["caption"])},
                        ],
                    }
                ]
                prompt_text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(
                    {
                        "prompt": prompt_text,
                        "multi_modal_data": {"video": (frames, metadata)},
                    }
                )
                valid.append(item)

            if not prompts:
                continue

            try:
                outputs = llm.generate(prompts, sampling)
            except Exception as exc:  # noqa: BLE001
                for item in valid:
                    results[item["clip_id"]] = {"score": None, "error": str(exc)}
                continue

            for item, out in zip(valid, outputs):
                content = out.outputs[0].text.strip()
                score, explanation = parse_response(content)
                if score is None:
                    results[item["clip_id"]] = {
                        "score": None,
                        "explanation": explanation,
                        "raw": content,
                        "error": "could_not_parse_score",
                    }
                else:
                    results[item["clip_id"]] = {
                        "score": score,
                        "explanation": explanation,
                        "raw": content,
                    }

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="VCInspector clip+caption verifier (vLLM)")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--model", type=str, default="dipta007/VCInspector-7B")
    parser.add_argument("--batch-size", type=int, default=16, help="clips per vLLM chunk")
    parser.add_argument("--video-max-pixels", type=int, default=DEFAULT_VIDEO_MAX_PIXELS)
    parser.add_argument("--num-frames", type=int, default=12, help="sampled frames per clip")
    parser.add_argument(
        "--fps-max-frames", type=int, default=None,
        help="alias for --num-frames (legacy; overrides --num-frames when set)",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="legacy; unused in vLLM mode")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    num_frames = int(args.fps_max_frames) if args.fps_max_frames else int(args.num_frames)

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        print("manifest must be a JSON list", file=sys.stderr)
        return 1

    results = run_vllm_inference(
        manifest,
        model=args.model,
        batch_size=args.batch_size,
        num_frames=num_frames,
        video_max_pixels=args.video_max_pixels,
        max_tokens=args.max_tokens,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} results -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
