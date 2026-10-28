"""YOLO26m via Ultralytics.

TODO [Team pipeline YOLO26m + preprocessing online]: ultralytics.YOLO('yolo26m.pt') and point data yaml at the offline export.
"""
from __future__ import annotations
import os
from typing import Any

def build_yolo26m(cfg: dict[str, Any]):
    """Build YOLO26m model safely across single-GPU and DDP ranks."""
    from ultralytics import YOLO

    from .weights import resolve_yolo_weights

    weights = resolve_yolo_weights(cfg)
    try:
        from ultralytics.utils.torch_utils import torch_distributed_zero_first
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        with torch_distributed_zero_first(local_rank):
            model = YOLO(weights, task="detect")
    except ImportError:
        model = YOLO(weights, task="detect")

    if model.task != "detect":
        raise RuntimeError(
            f"Expected detection task, got {model.task}"
        )

    return model