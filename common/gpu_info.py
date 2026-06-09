"""Print GPU usage banner at service start (visible in terminal)."""

from __future__ import annotations

import os
from typing import List, Optional


def resolve_gpu_ids(requested: List[int]) -> List[int]:
    try:
        import torch
    except ImportError:
        return [0]
    if not torch.cuda.is_available():
        return []
    n = torch.cuda.device_count()
    valid = [g for g in requested if 0 <= g < n]
    return valid if valid else list(range(n))


def log_service_gpus(
    service_id: str,
    service_name: str,
    model_label: str,
    requested_gpus: List[int],
    extra: Optional[str] = None,
) -> List[int]:
    """Print banner; return resolved GPU list actually used."""
    lines = [
        "",
        "=" * 72,
        f"[{service_id}] {service_name}",
        f"  Model: {model_label}",
    ]
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(all)")
    lines.append(f"  CUDA_VISIBLE_DEVICES: {cuda_visible}")

    try:
        import torch

        lines.append(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
        lines.append(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                lines.append(f"    cuda:{i} → {name} ({mem:.1f} GiB)")
    except Exception as exc:
        lines.append(f"  torch error: {exc}")

    resolved = resolve_gpu_ids(requested_gpus)
    lines.append(f"  Requested GPUs: {requested_gpus}")
    lines.append(f"  Using GPUs:     {resolved if resolved else 'CPU'}")
    if extra:
        lines.append(f"  Note: {extra}")
    lines.append("=" * 72)
    print("\n".join(lines), flush=True)
    return resolved
