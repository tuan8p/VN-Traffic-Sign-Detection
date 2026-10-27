"""Resolve YOLO26m publisher weights without downloading when Kaggle Model is attached."""
from __future__ import annotations

from pathlib import Path
from typing import Any

KAGGLE_YOLO26M_PT = Path(
    "/kaggle/input/models/ultralytics/yolo26/pytorch/default/1/yolo26m.pt"
)


def resolve_yolo_weights(cfg: dict[str, Any]) -> str:
    """Pick a local .pt file when present; otherwise fall back to Ultralytics weight name."""
    runtime = cfg.get("runtime_cfg") or {}
    for raw in (
        runtime.get("yolo_pretrained"),
        (cfg.get("model") or {}).get("weights"),
    ):
        if not isinstance(raw, str) or not raw.strip():
            continue
        path = Path(raw.strip())
        if path.is_file():
            return str(path.resolve())
    if KAGGLE_YOLO26M_PT.is_file():
        return str(KAGGLE_YOLO26M_PT.resolve())
    name = (cfg.get("model") or {}).get("weights", "yolo26m.pt")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model.weights must be a non-empty string")
    return name.strip()
