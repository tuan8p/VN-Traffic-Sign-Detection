"""YOLO26m via Ultralytics.

TODO [Team pipeline YOLO26m + preprocessing online]: ultralytics.YOLO('yolo26m.pt') and point data yaml at the offline export.
"""
from __future__ import annotations
from typing import Any

def build_yolo26m(cfg: dict[str, Any]):
    """Build YOLO26m model."""
    from ultralytics import YOLO

    from .weights import resolve_yolo_weights

    weights = resolve_yolo_weights(cfg)
    model = YOLO(weights, task="detect")

    if model.task != "detect":
        raise RuntimeError(
            f"Expected detection task, got {model.task}"
        )


    return model