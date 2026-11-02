"""Load a Faster R-CNN or YOLO26m checkpoint and run one image.

Dispatch on pipeline name; return boxes as list[dict] with keys:
  xyxy  : [x1, y1, x2, y2]  (pixel coords in the *resized* 640×640 space)
  score : float [0, 1]
  label : int (0-based class id, i.e. index into CLASS_NAMES)

When no trained checkpoint is available the module falls back to a
deterministic mock that produces plausible boxes for UI development.

Usage
-----
from demo.inference import predict, CLASS_NAMES
results = predict(pil_image, checkpoint="path/to/checkpoint_best.pt",
                  pipeline="faster_rcnn")
# results["boxes"] -> list[dict], results["inference_ms"] -> float
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

# ── Class names (52 VN traffic signs) ─────────────────────────────────────────
_CLASSES_CSV = Path(__file__).resolve().parents[1] / "configs" / "classes.csv"


def _load_class_names() -> list[str]:
    try:
        with _CLASSES_CSV.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        return [r["name_vi"] for r in rows]
    except Exception:
        return [f"class_{i}" for i in range(52)]


CLASS_NAMES: list[str] = _load_class_names()

# ── Constants ─────────────────────────────────────────────────────────────────
_IMG_SIZE = 640   # both models expect 640×640 input
_DEMO_SCORE_THRESHOLD = 0.25   # display threshold for demo (models use 0.001 during training)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pil_to_numpy_rgb(pil_image) -> np.ndarray:
    """Return uint8 H×W×3 RGB numpy array."""
    return np.array(pil_image.convert("RGB"))


def _letterbox(rgb: np.ndarray, size: int = _IMG_SIZE) -> tuple[np.ndarray, float, int, int]:
    """Letterbox to (size, size) with pad=114. Returns (img, scale, pad_x, pad_y)."""
    import cv2
    h, w = rgb.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def _boxes_to_original(boxes_xyxy, scale, pad_x, pad_y):
    """Map 640×640 box coords back to original image space."""
    out = []
    for x1, y1, x2, y2 in boxes_xyxy:
        ox1 = (x1 - pad_x) / scale
        oy1 = (y1 - pad_y) / scale
        ox2 = (x2 - pad_x) / scale
        oy2 = (y2 - pad_y) / scale
        out.append([ox1, oy1, ox2, oy2])
    return out


def _mock_predict(pil_image, pipeline: str, score_threshold: float = _DEMO_SCORE_THRESHOLD) -> dict[str, Any]:
    """Deterministic fake predictions for UI development (no checkpoint needed)."""
    import random as _rng
    rng = _rng.Random(42)  # fixed seed → stable boxes across reruns
    w, h = pil_image.size
    n = rng.randint(2, 6)
    boxes = []
    for _ in range(n):
        cx = rng.uniform(0.15, 0.85) * w
        cy = rng.uniform(0.15, 0.85) * h
        bw = rng.uniform(0.05, 0.20) * w
        bh = rng.uniform(0.05, 0.20) * h
        score = rng.uniform(score_threshold + 0.05, 0.99)
        label = rng.randint(0, len(CLASS_NAMES) - 1)
        boxes.append(dict(
            xyxy=[cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
            xyxy_640=None,   # N/A in mock
            score=round(score, 3),
            label=label,
            class_name=CLASS_NAMES[label],
        ))
    # slight variation between models so the two columns differ
    if pipeline == "yolo26m":
        for b in boxes[1:]:
            b["score"] = round(min(b["score"] * 1.05, 0.99), 3)
    return dict(
        boxes=boxes,
        inference_ms=round(150.0 + rng.uniform(0, 80) + (200 if pipeline == "faster_rcnn" else 0), 1),
        mode="mock",
        pipeline=pipeline,
        score_threshold=score_threshold,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Faster R-CNN inference
# ─────────────────────────────────────────────────────────────────────────────

def _infer_faster_rcnn(
    pil_image,
    checkpoint: str | Path,
    device: str = "cpu",
    score_threshold: float = _DEMO_SCORE_THRESHOLD,
) -> dict[str, Any]:
    import torch
    from vn_tsd.pipelines.faster_rcnn.model import load_checkpoint
    from torchvision.transforms.functional import to_tensor

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model, metadata = load_checkpoint(checkpoint, device=device)
    model.eval()

    # Prepare input: letterbox then normalise (model normalises internally)
    rgb = _pil_to_numpy_rgb(pil_image)
    lb, scale, pad_x, pad_y = _letterbox(rgb)
    tensor = to_tensor(lb).to(device)  # float32 [0,1] CHW

    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model([tensor])[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    raw_boxes = output["boxes"].cpu().tolist()
    raw_scores = output["scores"].cpu().tolist()
    raw_labels = output["labels"].cpu().tolist()

    boxes = []
    for xyxy_640, score, label_1based in zip(raw_boxes, raw_scores, raw_labels):
        if score < score_threshold:
            continue
        label = label_1based - 1  # convert 1-based → 0-based
        if not (0 <= label < len(CLASS_NAMES)):
            continue
        # coordinates in 640-space; also map back to original
        orig = _boxes_to_original([xyxy_640], scale, pad_x, pad_y)[0]
        boxes.append(dict(
            xyxy=orig,
            xyxy_640=list(xyxy_640),
            score=round(float(score), 4),
            label=label,
            class_name=CLASS_NAMES[label],
        ))

    return dict(
        boxes=boxes,
        inference_ms=round(elapsed_ms, 1),
        mode="real",
        pipeline="faster_rcnn",
        score_threshold=score_threshold,
        checkpoint=str(checkpoint),
    )


# ─────────────────────────────────────────────────────────────────────────────
# YOLO26m inference
# ─────────────────────────────────────────────────────────────────────────────

def _infer_yolo(
    pil_image,
    checkpoint: str | Path,
    device: str = "cpu",
    score_threshold: float = _DEMO_SCORE_THRESHOLD,
) -> dict[str, Any]:
    from ultralytics import YOLO
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = YOLO(str(checkpoint))
    w_orig, h_orig = pil_image.size

    t0 = time.perf_counter()
    results = model.predict(
        source=pil_image,
        imgsz=_IMG_SIZE,
        conf=score_threshold,
        device=device,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    boxes = []
    if results and len(results) > 0:
        det = results[0]
        if det.boxes is not None:
            for box in det.boxes:
                x1, y1, x2, y2 = box.xyxyn[0].tolist()  # normalised [0,1]
                # scale back to original pixel coords
                px1, py1, px2, py2 = x1 * w_orig, y1 * h_orig, x2 * w_orig, y2 * h_orig
                score = float(box.conf[0])
                label = int(box.cls[0])
                if not (0 <= label < len(CLASS_NAMES)):
                    continue
                boxes.append(dict(
                    xyxy=[px1, py1, px2, py2],
                    xyxy_640=None,
                    score=round(score, 4),
                    label=label,
                    class_name=CLASS_NAMES[label],
                ))

    return dict(
        boxes=boxes,
        inference_ms=round(elapsed_ms, 1),
        mode="real",
        pipeline="yolo26m",
        score_threshold=score_threshold,
        checkpoint=str(checkpoint),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    pil_image,
    checkpoint: str | Path | None = None,
    pipeline: str = "faster_rcnn",
    device: str = "cpu",
    score_threshold: float = _DEMO_SCORE_THRESHOLD,
) -> dict[str, Any]:
    """Run inference on a PIL Image; return structured result dict.

    Parameters
    ----------
    pil_image   : PIL.Image.Image
    checkpoint  : path to checkpoint_best.pt; if None or missing, falls back to mock.
    pipeline    : "faster_rcnn" or "yolo26m"
    device      : "cpu" | "cuda" | "cuda:0" etc.
    score_threshold : minimum confidence to include a box in output

    Returns
    -------
    dict with keys:
        boxes          : list[dict]  — each has xyxy, score, label, class_name
        inference_ms   : float
        mode           : "real" | "mock"
        pipeline       : str
        score_threshold: float
    """
    ckpt = Path(checkpoint) if checkpoint else None

    # ── Fall back to mock if no checkpoint ────────────────────────────────────
    if ckpt is None or not ckpt.is_file():
        return _mock_predict(pil_image, pipeline=pipeline, score_threshold=score_threshold)

    try:
        if pipeline == "faster_rcnn":
            return _infer_faster_rcnn(pil_image, ckpt, device=device, score_threshold=score_threshold)
        elif pipeline == "yolo26m":
            return _infer_yolo(pil_image, ckpt, device=device, score_threshold=score_threshold)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline!r}. Choose 'faster_rcnn' or 'yolo26m'.")
    except Exception as exc:
        # Surface error in UI-friendly way, still return mock so demo is usable
        import traceback
        return dict(
            boxes=[],
            inference_ms=0.0,
            mode="error",
            pipeline=pipeline,
            score_threshold=score_threshold,
            error=str(exc),
            traceback=traceback.format_exc(),
        )


def draw_boxes(pil_image, result: dict[str, Any], *, line_width: int = 2) -> Any:
    """Draw bounding boxes from a predict() result onto a PIL Image copy.

    Returns the annotated PIL Image.
    """
    from PIL import ImageDraw, ImageFont
    import colorsys

    img = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    # Assign a colour per class (hue-based, consistent across calls)
    def _class_color(label: int) -> tuple[int, int, int]:
        hue = (label * 37) % 360 / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        return int(r * 255), int(g * 255), int(b * 255)

    try:
        font = ImageFont.truetype("arial.ttf", size=13)
    except Exception:
        font = ImageFont.load_default()

    for det in result.get("boxes", []):
        x1, y1, x2, y2 = det["xyxy"]
        color = _class_color(det["label"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        label_text = f"{det['class_name']} {det['score']:.2f}"
        # Draw text background
        bbox_text = draw.textbbox((x1, y1 - 14), label_text, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((x1, y1 - 14), label_text, fill="white", font=font)

    return img
