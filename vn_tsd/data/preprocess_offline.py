"""Shared offline preprocessing for detection.

Keeps images + YOLO boxes (no classification crop). Writes a processed tree
both pipelines can read:

    data/processed/
      images/{train,val,test}/
      labels/{train,val,test}/
      data.yaml
      meta.json

TODO [Team analysis + preprocessing offline]: discover VNTS layout, drop tiny boxes, letterbox to a shared size,
write YOLO txt, emit data.yaml (nc, names from configs/classes.csv).
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

def run_offline(cfg: dict[str, Any], data_root: str | Path) -> Path:
    raise NotImplementedError(
        f"TODO [Team analysis + preprocessing offline]: offline preprocess from {data_root} using shared image size "
        f"{cfg.get('preprocess', {}).get('image_size')}"
    )
