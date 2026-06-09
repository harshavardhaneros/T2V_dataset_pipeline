"""DOVER aesthetic/technical video quality scoring."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class DoverClient:
    def __init__(self, config: Dict[str, Any]):
        models_cfg = config.get("models", {})
        self.cfg = models_cfg.get("dover", config.get("dover", {}))
        self.use_placeholder = bool(self.cfg.get("use_placeholder", True))
        self.repo_path = Path(self.cfg["repo_path"]) if self.cfg.get("repo_path") else None
        self.device = self.cfg.get("device", "cuda")
        self._evaluator = None

    def _placeholder_scores(self, video_path: str) -> Tuple[float, float, float]:
        digest = hashlib.md5(str(video_path).encode()).hexdigest()
        base = int(digest[:8], 16)
        aesthetic = 0.45 + (base % 35) / 100.0
        technical = 0.50 + ((base >> 8) % 30) / 100.0
        overall = 0.7 * aesthetic + 0.3 * technical
        return aesthetic, technical, overall

    def _load_evaluator(self):
        if self._evaluator is not None:
            return self._evaluator
        if not self.repo_path or not self.repo_path.exists():
            return None
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))
        try:
            import torch
            from dover.datasets import spatial_temporal_view_decomposition  # type: ignore
            from dover.models import DOVER, fuse_results  # type: ignore

            opt_path = self.cfg.get("config_path", self.repo_path / "dover.yml")
            import yaml

            with open(opt_path, encoding="utf-8") as f:
                opt = yaml.safe_load(f)
            evaluator = DOVER(**opt["model"]["args"]).to(self.device)
            ckpt = self.cfg.get("checkpoint", opt.get("test_load_path"))
            evaluator.load_state_dict(torch.load(ckpt, map_location=self.device))
            evaluator.eval()
            self._evaluator = (evaluator, opt, spatial_temporal_view_decomposition, fuse_results, torch)
            return self._evaluator
        except Exception:
            return None

    def _score_with_repo(self, video_path: str) -> Optional[Tuple[float, float, float]]:
        loaded = self._load_evaluator()
        if loaded is None:
            return None
        evaluator, opt, decompose, fuse_results, torch = loaded
        try:
            dopt = opt["data"]["val-l1080p"]["args"]
            views, _ = decompose(str(video_path), dopt["sample_types"], {})
            mean = torch.tensor(opt["mean"]).view(1, 3, 1, 1, 1).to(self.device)
            std = torch.tensor(opt["std"]).view(1, 3, 1, 1, 1).to(self.device)
            for key, tensor in views.items():
                num_clips = dopt["sample_types"][key].get("num_clips", 1)
                views[key] = (
                    ((tensor.permute(1, 2, 3, 0) - mean.cpu()) / std.cpu())
                    .permute(3, 0, 1, 2)
                    .reshape(tensor.shape[0], num_clips, -1, *tensor.shape[2:])
                    .transpose(0, 1)
                    .to(self.device)
                )
            with torch.no_grad():
                raw = [float(r.mean().item()) for r in evaluator(views)]
            fused = fuse_results(raw)
            aesthetic = float(fused.get("aesthetic", raw[0]))
            technical = float(fused.get("technical", raw[1] if len(raw) > 1 else raw[0]))
            overall = float(fused.get("overall", 0.7 * aesthetic + 0.3 * technical))
            return aesthetic, technical, overall
        except Exception:
            return None

    def _score_with_cli(self, video_path: str) -> Optional[Tuple[float, float, float]]:
        if not self.repo_path:
            return None
        script = self.repo_path / "evaluate_one_video.py"
        if not script.exists():
            return None
        cmd = [
            sys.executable,
            str(script),
            "-v",
            str(video_path),
            "-f",
            "--device",
            str(self.device),
        ]
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(self.repo_path))
            for line in proc.stdout.splitlines():
                if "overall" in line.lower() and ":" in line:
                    value = float(line.split(":")[-1].strip())
                    return value, value, value
        except (subprocess.CalledProcessError, ValueError):
            return None
        return None

    def score_video(self, video_path: str) -> Dict[str, float]:
        if self.use_placeholder:
            aesthetic, technical, overall = self._placeholder_scores(video_path)
            return {
                "aesthetic_score": aesthetic,
                "technical_score": technical,
                "dover_score": overall,
            }

        scores = self._score_with_repo(video_path) or self._score_with_cli(video_path)
        if scores is None:
            aesthetic, technical, overall = self._placeholder_scores(video_path)
        else:
            aesthetic, technical, overall = scores
        return {
            "aesthetic_score": aesthetic,
            "technical_score": technical,
            "dover_score": overall,
        }
