"""YOLO26m via Ultralytics.

TODO [Team pipeline YOLO26m + preprocessing online]: ultralytics.YOLO('yolo26m.pt') and point data yaml at the offline export.
"""
from __future__ import annotations
from typing import Any

from ultralytics import YOLO

def build_yolo26m(cfg: dict[str, Any]):
    """Build YOLO26m model.

    Args:
        cfg: Config dict.

    Returns:
        YOLO model.
    """
    model_cfg = cfg.get("model", {})
    weights = model_cfg.get("weights", "yolo26m.pt")

    if not isinstance(weights, str) or not weights.strip():
        raise ValueError(
            "model.weights must be a non-empty string"
        )

    model = YOLO(weights, task="detect")

    if model.task != "detect":
        raise RuntimeError(
            f"Expected detection task, got {model.task}"
        )


    return model