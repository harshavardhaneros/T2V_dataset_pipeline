"""Service 13: VCInspector caption verification (clip MP4 + caption)."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.base_service import BaseService
from common.caption_text import prose_caption_for_export
from common.gpu_info import resolve_gpu_ids
from common.metadata_manager import MetadataManager
from common.progress import iter_progress
from common.qwen_video_caption import ensure_clip_mp4


def _caption_text(rec: Dict[str, Any]) -> str:
    text = prose_caption_for_export(rec)
    if text:
        return text
    return str(rec.get("generated_caption") or rec.get("caption") or "").strip()


class CaptionVerifyService(BaseService):
    service_id = "s13"
    service_name = "s13_caption_verify"
    owned_fields = [
        "caption_verify_score",
        "caption_verify_score_norm",
        "caption_verify_explanation",
        "caption_verify_pass",
    ]

    def _s13_cfg(self) -> Dict[str, Any]:
        return self.config.get("pipeline", {}).get("s13", {})

    def _min_score(self) -> int:
        gate = self.config.get("thresholds", {}).get("caption_verify", {})
        return int(gate.get("min_score", 4))

    def _cuda_visible(self, gpu_ids: List[int]) -> str:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible:
            return ",".join(str(g) for g in gpu_ids)
        parts = [p.strip() for p in visible.split(",") if p.strip()]
        mapped = []
        for g in gpu_ids:
            if 0 <= g < len(parts):
                mapped.append(parts[g])
            else:
                mapped.append(str(g))
        return ",".join(mapped)

    def _run_worker(
        self,
        manifest_path: Path,
        results_path: Path,
        *,
        gpu_ids: List[int],
    ) -> None:
        cfg = self._s13_cfg()
        worker = cfg.get("worker", "scripts/vcinspector_worker.py")
        worker_path = Path(worker)
        if not worker_path.is_absolute():
            worker_path = self.pipeline_root / worker_path
        if not worker_path.exists():
            raise FileNotFoundError(f"VCInspector worker not found: {worker_path}")

        conda_env = cfg.get("conda_env", "vcinspector")
        conda_base = os.environ.get("CONDA_EXE", "conda")
        if conda_base.endswith("/conda"):
            conda_sh = str(Path(conda_base).parent / "etc" / "profile.d" / "conda.sh")
        else:
            conda_sh = "/opt/conda/etc/profile.d/conda.sh"
        if not Path(conda_sh).exists():
            import subprocess as sp

            conda_sh = sp.check_output(
                ["conda", "info", "--base"], text=True
            ).strip() + "/etc/profile.d/conda.sh"

        model_id = str(cfg.get("model_id", "dipta007/VCInspector-7B"))
        local_model = self.config.get("models", {}).get("vcinspector", {}).get("model_path")
        if local_model and Path(local_model).exists():
            model_id = str(local_model)

        py_cmd = " ".join(
            [
                f"python {worker_path}",
                f"--manifest {manifest_path}",
                f"--out {results_path}",
                f"--model {model_id}",
                f"--batch-size {int(cfg.get('batch_size', 1))}",
                f"--video-max-pixels {int(cfg.get('video_max_pixels', 151200))}",
                f"--fps {float(cfg.get('fps', 1.0))}",
                f"--fps-max-frames {int(cfg.get('fps_max_frames', 12))}",
                f"--max-tokens {int(cfg.get('max_tokens', 256))}",
            ]
        )
        bash_cmd = (
            f"source {conda_sh} && conda activate {conda_env} && {py_cmd}"
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self._cuda_visible(gpu_ids)
        env["USE_HF"] = "1"
        env.setdefault("VIDEO_MAX_PIXELS", str(cfg.get("video_max_pixels", 151200)))
        env.setdefault("FPS_MAX_FRAMES", str(cfg.get("fps_max_frames", 12)))

        subprocess.run(["bash", "-lc", bash_cmd], check=True, env=env)

    def process_movie(self) -> Dict[str, Any]:
        cfg = self._s13_cfg()
        if not cfg.get("enabled", True):
            return {"skipped": True, "reason": "s13 disabled"}

        records = self.metadata.read_all()
        if self.should_skip_movie(records):
            return {"skipped": True, "reason": "all clips done"}

        min_score = self._min_score()
        gpu_ids = resolve_gpu_ids([int(g) for g in cfg.get("gpu_ids", [0])])
        clips_dir = self.movie_dir / "clips"
        allowed_verdicts = set(
            cfg.get("include_verdicts", ["FINAL", "REVIEW"])
        )

        test_max = self.config.get("_test", {}).get("max_clips")
        s13_max = cfg.get("max_clips")
        cap = int(s13_max) if s13_max else (int(test_max) if test_max else None)

        manifest: List[Dict[str, str]] = []
        skipped_no_caption = 0
        skipped_no_clip = 0

        for rec in records:
            if self.should_skip_clip(rec):
                continue
            if not rec.get("keep", True) or rec.get("reject"):
                continue
            if rec.get("verdict") not in allowed_verdicts:
                continue
            caption = _caption_text(rec)
            if not caption:
                skipped_no_caption += 1
                rec["caption_verify_score"] = None
                rec["caption_verify_score_norm"] = None
                rec["caption_verify_explanation"] = ""
                rec["caption_verify_pass"] = False
                MetadataManager.mark_done(rec, self.service_id)
                continue
            if cap is not None and len(manifest) >= cap:
                continue

            if not self.movie_video:
                raise FileNotFoundError(f"No movie video in {self.movie_dir}")
            clip_path = ensure_clip_mp4(
                self.movie_video, rec, clips_dir, self.config
            )
            if not clip_path or not clip_path.exists():
                skipped_no_clip += 1
                rec["caption_verify_score"] = None
                rec["caption_verify_score_norm"] = None
                rec["caption_verify_explanation"] = "clip_mp4_missing"
                rec["caption_verify_pass"] = False
                MetadataManager.mark_done(rec, self.service_id)
                continue

            manifest.append(
                {
                    "clip_id": rec["clip_id"],
                    "caption": caption,
                    "clip_path": str(clip_path.resolve()),
                }
            )

        if not manifest:
            self.metadata.write_all(records)
            return {
                "verified": 0,
                "skipped_no_caption": skipped_no_caption,
                "skipped_no_clip": skipped_no_clip,
                "mean_score": 0,
                "pass_rate": 0,
            }

        scratch = self.movie_dir / "scratch" / "s13"
        scratch.mkdir(parents=True, exist_ok=True)
        manifest_path = scratch / "vcinspector_manifest.json"
        results_path = scratch / "vcinspector_results.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        self._run_worker(manifest_path, results_path, gpu_ids=gpu_ids)
        raw_results: Dict[str, Any] = json.loads(
            results_path.read_text(encoding="utf-8")
        )

        verified = passed = 0
        scores: List[float] = []
        by_id = {rec["clip_id"]: rec for rec in records}

        for item in manifest:
            rec = by_id.get(item["clip_id"])
            if not rec:
                continue
            row = raw_results.get(item["clip_id"], {})
            score = row.get("score")
            explanation = str(row.get("explanation") or row.get("error") or "")
            if score is not None:
                score = int(score)
                score = max(1, min(5, score))
                rec["caption_verify_score"] = score
                rec["caption_verify_score_norm"] = round((score - 1) / 4.0, 3)
                rec["caption_verify_explanation"] = explanation
                rec["caption_verify_pass"] = score >= min_score
                scores.append(float(score))
                if rec["caption_verify_pass"]:
                    passed += 1
                verified += 1
            else:
                rec["caption_verify_score"] = None
                rec["caption_verify_score_norm"] = None
                rec["caption_verify_explanation"] = explanation or "verify_failed"
                rec["caption_verify_pass"] = False
            MetadataManager.mark_done(rec, self.service_id)

        self.metadata.write_all(records)
        mean_score = sum(scores) / len(scores) if scores else 0.0
        pass_rate = passed / verified if verified else 0.0
        return {
            "verified": verified,
            "skipped_no_caption": skipped_no_caption,
            "skipped_no_clip": skipped_no_clip,
            "mean_score": round(mean_score, 3),
            "pass_rate": round(pass_rate, 3),
            "min_score": min_score,
            "model": cfg.get("model_id", "dipta007/VCInspector-7B"),
            "gpus": gpu_ids,
            "input_mode": "native_mp4",
        }
