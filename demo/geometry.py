"""Geometric and scene-structure analysis for the demo.

Owner: Team demo + phân tích hình học và cấu trúc cảnh

Nguồn: mục 3.2, CV-project-tonghop.pdf. Hướng đã chọn:
  Phát hiện biên (Canny) + đường thẳng (HoughLinesP)

Pipeline:
  1. Grayscale → Gaussian blur → Canny edge detection
  2. Probabilistic Hough Transform tìm line segments
  3. Vẽ edges và lines lên overlay, trả metadata

Returns kết quả để demo/app.py vẽ lên ảnh.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ─── Tuning constants ──────────────────────────────────────────────────────────
_BLUR_KERNEL = 5           # Gaussian blur kernel size (odd)
_CANNY_LO = 50             # Canny lower threshold
_CANNY_HI = 150            # Canny upper threshold
_HOUGH_RHO = 1             # Distance resolution (px)
_HOUGH_THETA = np.pi / 180 # Angle resolution (rad)
_HOUGH_THRESHOLD = 80      # Accumulator threshold
_HOUGH_MIN_LINE = 40       # Min line length (px)
_HOUGH_MAX_GAP = 10        # Max gap in a line (px)
_EDGE_COLOR = (0, 255, 0)  # BGR green for edges overlay
_LINE_COLOR = (0, 0, 255)  # BGR red for Hough lines
_LINE_THICKNESS = 2


def _load_bgr(image_input) -> np.ndarray:
    """Accept file path, bytes or numpy array; always return uint8 BGR H×W×3."""
    if isinstance(image_input, np.ndarray):
        img = image_input
    elif isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_input}")
    elif isinstance(image_input, (bytes, bytearray)):
        arr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image bytes")
    else:
        raise TypeError(f"Unsupported input type: {type(image_input)}")

    if img.ndim == 2:  # grayscale → BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # BGRA → BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def analyze_scene(
    image_input,
    *,
    canny_lo: int = _CANNY_LO,
    canny_hi: int = _CANNY_HI,
    blur_kernel: int = _BLUR_KERNEL,
    hough_threshold: int = _HOUGH_THRESHOLD,
    hough_min_line: int = _HOUGH_MIN_LINE,
    hough_max_gap: int = _HOUGH_MAX_GAP,
) -> dict[str, Any]:
    """Run Canny edge detection + Probabilistic Hough Transform.

    Parameters
    ----------
    image_input : str | Path | bytes | np.ndarray
        Input image.  PIL Images should be converted by the caller
        (``np.array(pil_img)`` in RGB, will be handled internally).

    Returns
    -------
    dict with keys:
        edge_overlay  : np.ndarray uint8 H×W×3 BGR  — edges highlighted in green
        line_overlay  : np.ndarray uint8 H×W×3 BGR  — lines on original image
        combined_overlay : np.ndarray                — edges + lines together
        edges         : np.ndarray uint8 H×W        — binary Canny mask
        lines         : list[tuple[int,int,int,int]] — [(x1,y1,x2,y2), ...]
        stats : dict  — num_edge_px, num_lines, edge_density_pct, mean_line_length,
                        canny_lo, canny_hi, hough_threshold
    """
    # ── 1. Load ─────────────────────────────────────────────────────────────────
    bgr = _load_bgr(image_input)
    h, w = bgr.shape[:2]

    # ── 2. Preprocessing ────────────────────────────────────────────────────────
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # ── 3. Canny ────────────────────────────────────────────────────────────────
    edges = cv2.Canny(blurred, canny_lo, canny_hi)

    # ── 4. Hough Lines ──────────────────────────────────────────────────────────
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=_HOUGH_RHO,
        theta=_HOUGH_THETA,
        threshold=hough_threshold,
        minLineLength=hough_min_line,
        maxLineGap=hough_max_gap,
    )
    lines: list[tuple[int, int, int, int]] = []
    if raw_lines is not None:
        for seg in raw_lines:
            # cv2 <5: shape (N,1,4); cv2 >=5: shape (N,4) — handle both
            pts = seg[0] if seg.ndim == 2 else seg
            x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
            lines.append((x1, y1, x2, y2))

    # ── 5. Overlays ─────────────────────────────────────────────────────────────
    # Edge overlay: green channels on grayscale background
    edge_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edge_overlay[edges > 0] = _EDGE_COLOR

    # Line overlay: lines drawn on original colour image
    line_overlay = bgr.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(line_overlay, (x1, y1), (x2, y2), _LINE_COLOR, _LINE_THICKNESS)

    # Combined: edge mask + lines on the original
    combined_overlay = bgr.copy()
    combined_overlay[edges > 0] = _EDGE_COLOR
    for x1, y1, x2, y2 in lines:
        cv2.line(combined_overlay, (x1, y1), (x2, y2), _LINE_COLOR, _LINE_THICKNESS)

    # ── 6. Stats ────────────────────────────────────────────────────────────────
    num_edge_px = int(np.count_nonzero(edges))
    total_px = h * w
    edge_density = round(100.0 * num_edge_px / total_px, 2) if total_px else 0.0

    lengths = []
    for x1, y1, x2, y2 in lines:
        lengths.append(float(np.hypot(x2 - x1, y2 - y1)))
    mean_len = round(float(np.mean(lengths)), 1) if lengths else 0.0

    stats = dict(
        num_edge_px=num_edge_px,
        num_lines=len(lines),
        edge_density_pct=edge_density,
        mean_line_length=mean_len,
        image_size=(w, h),
        canny_lo=canny_lo,
        canny_hi=canny_hi,
        hough_threshold=hough_threshold,
    )

    return dict(
        edge_overlay=edge_overlay,
        line_overlay=line_overlay,
        combined_overlay=combined_overlay,
        edges=edges,
        lines=lines,
        stats=stats,
    )


def overlay_on_pil(pil_image, result_key: str = "combined_overlay", **kwargs):
    """Convenience: accept a PIL Image, return a PIL Image.

    result_key : one of 'edge_overlay', 'line_overlay', 'combined_overlay'
    """
    from PIL import Image as PILImage
    arr_rgb = np.array(pil_image.convert("RGB"))
    arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    out = analyze_scene(arr_bgr, **kwargs)
    overlay_bgr = out[result_key]
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(overlay_rgb), out["stats"]
