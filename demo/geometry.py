"""Geometric and scene-structure analysis for the demo.

Owner: Team demo + phân tích hình học và cấu trúc cảnh

Nguồn: mục 3.2, CV-project-tonghop.pdf. Hướng đã chọn:
  1. Phát hiện biên (Canny) + đường thẳng (HoughLinesP)
  2. Ước lượng phép biến đổi phối cảnh 2D Homography để nắn thẳng (rectify) biển báo

Pipeline:
  Phần A - Toàn cảnh:
    1. Grayscale -> Gaussian blur -> Canny edge detection
    2. Probabilistic Hough Transform tìm line segments
    3. Vẽ edges và lines lên overlay, trả metadata
  Phần B - Nắn thẳng biển báo (Perspective Rectification):
    1. Trích xuất vùng ROI chứa biển báo (có padding an toàn)
    2. Dò tìm 4 đỉnh góc của biển báo (đa giác hoặc hình chữ nhật nghiêng tối thiểu)
    3. Tính ma trận Homography H (3x3) với cv2.getPerspectiveTransform
    4. Warp biến đổi phối cảnh đưa biển báo về góc nhìn trực diện chính diện
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


# ─────────────────────────────────────────────────────────────────────────────
# 3.2.B - Nắn thẳng phối cảnh biển báo (Homography Rectification)
# ─────────────────────────────────────────────────────────────────────────────

def order_points(pts: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 tọa độ theo chiều kim đồng hồ: [trên-trái, trên-phải, dưới-phải, dưới-trái].
    pts shape: (4, 2)
    """
    rect = np.zeros((4, 2), dtype="float32")
    # Điểm trên-trái có tổng x + y nhỏ nhất; dưới-phải có tổng lớn nhất
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Điểm trên-phải có hiệu y - x nhỏ nhất; dưới-trái có hiệu lớn nhất
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def rectify_sign(
    image_input,
    bbox: list[float] | tuple[float, float, float, float],
    target_size: tuple[int, int] = (240, 240),
    padding: float = 0.15,
) -> dict[str, Any]:
    """Nắn thẳng biển báo bị nghiêng / xiên góc về dạng chính diện (frontal view).

    Thuật toán:
      1. Mở rộng bounding box thêm tỷ lệ padding để không bị mất góc viền biển báo.
      2. Cắt ROI, xử lý mờ Gaussian, dò biên Canny và tìm contour lớn nhất trong ROI.
      3. Xác định 4 đỉnh góc của biển báo (đa giác xấp xỉ hoặc khung chữ nhật nghiêng tối thiểu).
      4. Tính ma trận Homography H qua cv2.getPerspectiveTransform.
      5. Thực hiện phép biến đổi cv2.warpPerspective để tái tạo ảnh chính diện chuẩn.

    Parameters
    ----------
    image_input : str | Path | np.ndarray
        Ảnh gốc.
    bbox : [x1, y1, x2, y2]
        Tọa độ bounding box trong không gian ảnh gốc.
    target_size : (width, height)
        Kích thước ảnh sau khi nắn thẳng (mặc định 240x240).
    padding : float
        Tỷ lệ mở rộng biên vùng ROI (mặc định 15%).

    Returns
    -------
    dict:
        rectified_bgr : np.ndarray uint8 - Ảnh biển báo sau khi nắn thẳng trực diện
        roi_annotated_bgr : np.ndarray uint8 - Vùng ROI gốc có vẽ 4 đỉnh góc và khung tứ giác
        roi_crop_bgr : np.ndarray uint8 - Vùng ROI gốc chưa vẽ
        homography_matrix : np.ndarray (3, 3) float64 - Ma trận biến đổi phối cảnh 2D Homography H
        src_points_global : np.ndarray (4, 2) - 4 điểm góc nguồn trên ảnh gốc
        src_points_roi : np.ndarray (4, 2) - 4 điểm góc nguồn trên ROI
        dst_points : np.ndarray (4, 2) - 4 điểm góc đích chuẩn
        method : str - Phương pháp xác định 4 đỉnh góc
    """
    bgr = _load_bgr(image_input)
    h, w = bgr.shape[:2]

    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    # Giới hạn tọa độ hợp lệ
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(w), x2), min(float(h), y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Tọa độ bounding box không hợp lệ: {bbox}")

    pw = (x2 - x1) * padding
    ph = (y2 - y1) * padding
    rx1 = max(0, int(round(x1 - pw)))
    ry1 = max(0, int(round(y1 - ph)))
    rx2 = min(w, int(round(x2 + pw)))
    ry2 = min(h, int(round(y2 + ph)))

    roi = bgr[ry1:ry2, rx1:rx2].copy()
    if roi.size == 0:
        raise ValueError("Vùng ROI cắt ra có kích thước 0")

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chosen_pts = None
    method = "Khung bounding box trực tiếp"

    if contours:
        roi_area = (rx2 - rx1) * (ry2 - ry1)
        valid_cnts = [c for c in contours if cv2.contourArea(c) > 0.05 * roi_area]
        if valid_cnts:
            largest = max(valid_cnts, key=cv2.contourArea)
            peri = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, 0.04 * peri, True)
            if len(approx) == 4:
                chosen_pts = approx.reshape(4, 2).astype("float32")
                method = "Xấp xỉ đa giác 4 đỉnh (Polygon Approx)"
            else:
                rect = cv2.minAreaRect(largest)
                box = cv2.boxPoints(rect)
                chosen_pts = np.array(box, dtype="float32")
                method = "Khung tứ giác nghiêng tối thiểu (MinArea Quad)"

    if chosen_pts is None:
        chosen_pts = np.array([
            [x1 - rx1, y1 - ry1],
            [x2 - rx1, y1 - ry1],
            [x2 - rx1, y2 - ry1],
            [x1 - rx1, y2 - ry1],
        ], dtype="float32")

    ordered_roi_pts = order_points(chosen_pts)
    ordered_global_pts = ordered_roi_pts + np.array([rx1, ry1], dtype="float32")

    tw, th = target_size
    dst_pts = np.array([
        [0, 0],
        [tw - 1, 0],
        [tw - 1, th - 1],
        [0, th - 1],
    ], dtype="float32")

    try:
        H = cv2.getPerspectiveTransform(ordered_global_pts, dst_pts)
        rectified = cv2.warpPerspective(bgr, H, (tw, th), flags=cv2.INTER_LANCZOS4)
    except Exception:
        # Fallback to direct ROI crop and resize if matrix estimation fails
        rectified = cv2.resize(roi, (tw, th), interpolation=cv2.INTER_LINEAR)
        H = np.eye(3, dtype="float32")
        method = "Khung chữ nhật trực tiếp (Fallback)" 

    # Vẽ các đỉnh và đường nối lên ROI minh họa
    roi_annotated = roi.copy()
    pts_int = ordered_roi_pts.astype(np.int32)
    cv2.polylines(roi_annotated, [pts_int], isClosed=True, color=(0, 255, 0), thickness=2)
    for i, (px, py) in enumerate(pts_int):
        cv2.circle(roi_annotated, (px, py), 4, (0, 0, 255), -1)
        cv2.putText(roi_annotated, str(i + 1), (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    return {
        "rectified_bgr": rectified,
        "roi_annotated_bgr": roi_annotated,
        "roi_crop_bgr": roi,
        "homography_matrix": H,
        "src_points_global": ordered_global_pts,
        "src_points_roi": ordered_roi_pts,
        "dst_points": dst_pts,
        "method": method,
        "target_size": (tw, th),
    }


def rectify_sign_pil(pil_image, bbox: list[float] | tuple[float, float, float, float], **kwargs) -> dict[str, Any]:
    """Hàm tiện ích nhận PIL Image và trả về các kết quả dưới dạng PIL Image."""
    from PIL import Image as PILImage
    arr_rgb = np.array(pil_image.convert("RGB"))
    arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)

    res = rectify_sign(arr_bgr, bbox, **kwargs)

    rect_rgb = cv2.cvtColor(res["rectified_bgr"], cv2.COLOR_BGR2RGB)
    annot_rgb = cv2.cvtColor(res["roi_annotated_bgr"], cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.cvtColor(res["roi_crop_bgr"], cv2.COLOR_BGR2RGB)

    return {
        "rectified_pil": PILImage.fromarray(rect_rgb),
        "roi_annotated_pil": PILImage.fromarray(annot_rgb),
        "roi_crop_pil": PILImage.fromarray(crop_rgb),
        "homography_matrix": res["homography_matrix"],
        "method": res["method"],
        "src_points": res["src_points_global"],
    }
