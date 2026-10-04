"""Geometric and scene-structure analysis for the demo.

Owner: Team demo + phân tích hình học và cấu trúc cảnh

Nguồn: mục 3.2, CV-project-tonghop.pdf. Làm ít nhất một hướng và ghi rõ hướng đã chọn.

TODO [Team demo + phân tích hình học và cấu trúc cảnh]:
  - Phát hiện biên, đường thẳng hoặc góc (edges, lines, corners)
  - Ước lượng affine hoặc projective
  - Ghép ảnh (panorama) hoặc căn chỉnh nhiều ảnh cùng một cảnh
  - Ước lượng mặt phẳng hoặc vùng bề mặt chính

Trả về kết quả để demo/app.py vẽ lên ảnh.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

def analyze_scene(image_path: str | Path) -> dict[str, Any]:
    raise NotImplementedError(
        "TODO [Team demo + phân tích hình học và cấu trúc cảnh]: geometric / scene-structure analysis for " + str(image_path)
    )
