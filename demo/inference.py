"""Load a Faster R-CNN or YOLO26m checkpoint and run one image.

TODO [Team demo + phân tích hình học và cấu trúc cảnh]: dispatch on pipeline name, return boxes as xyxy + score + class id.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

def predict(image_path: str | Path, checkpoint: str | Path, pipeline: str) -> list[dict[str, Any]]:
    raise NotImplementedError(f"TODO [Team demo + phân tích hình học và cấu trúc cảnh]: demo inference for {pipeline} from {checkpoint} on {image_path}")
