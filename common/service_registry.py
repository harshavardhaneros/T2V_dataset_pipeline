"""Dynamic service registry — no hardcoded pipeline imports."""

from __future__ import annotations

import importlib
from typing import Dict, Type

from common.base_service import BaseService

SERVICE_MODULES = {
    "s1": ("services.service_01_extract.service", "ExtractService"),
    "s2": ("services.service_02_dedup.service", "DedupService"),
    "s3": ("services.service_03_band_removal.service", "BandRemovalService"),
    "s4": ("services.service_04_watermark.service", "WatermarkService"),
    "s5": ("services.service_05_classify.service", "ClassifyService"),
    "s6": ("services.service_06_verify.service", "VerifyService"),
    "s7": ("services.service_07_actor_tagging.service", "ActorTaggingService"),
    "s8": ("services.service_08_caption.service", "CaptionService"),
    "s9": ("services.service_09_quality_scoring.service", "QualityScoringService"),
    "s10": ("services.service_10_gate.service", "GateService"),
    "s11": ("services.service_11_export.service", "ExportService"),
    "s12": ("services.service_12_report.service", "ReportService"),
}


def _load_class(module_path: str, class_name: str) -> Type[BaseService]:
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def get_service_class(step_id: str) -> Type[BaseService]:
    if step_id not in SERVICE_MODULES:
        raise KeyError(f"Unknown service: {step_id}")
    module_path, class_name = SERVICE_MODULES[step_id]
    return _load_class(module_path, class_name)


def build_registry() -> Dict[str, Type[BaseService]]:
    return {sid: get_service_class(sid) for sid in SERVICE_MODULES}
