"""Test/val evaluation with the same COCOeval path as Faster R-CNN."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from vn_tsd.eval.coco_eval import coco_metrics, detection_thresholds, evaluator_metadata
from vn_tsd.eval.metrics import summarize_yolo
from vn_tsd.pipelines.faster_rcnn.dataset import load_cache


def _device_arg(cfg: dict[str, Any]):
    runtime = cfg.get("runtime_cfg", {})
    device = runtime.get("device", "cuda")
    if device == "cuda":
        return 0
    return device


def _predictions_coco(model, dataset, thresholds: dict[str, float | int]) -> list[dict]:
    paths = [str(record["path"]) for record in dataset.records]
    if not paths:
        return []
    results = model.predict(
        source=paths,
        imgsz=640,
        conf=thresholds["score_thresh"],
        iou=thresholds["nms_thresh"],
        max_det=thresholds["detections_per_img"],
        verbose=False,
    )
    if len(results) != len(dataset.records):
        raise ValueError("YOLO prediction count mismatch")
    predictions: list[dict] = []
    for result, record in zip(results, dataset.records):
        image_id = int(record["image_id"])
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        labels = result.boxes.cls.cpu().tolist()
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            cls = int(label)
            if not 0 <= cls < 52 or not 0.0 <= float(score) <= 1.0 or x2 <= x1 or y2 <= y1:
                raise ValueError("Invalid YOLO prediction for COCO export")
            predictions.append(
                dict(
                    image_id=image_id,
                    category_id=cls,
                    bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    score=float(score),
                )
            )
    return predictions


def evaluate_split_coco(cfg: dict[str, Any], checkpoint: Path, split: str = "test") -> dict[str, Any]:
    from ultralytics import YOLO

    sets, _ = load_cache(cfg, splits=(split,))
    dataset = sets[split]
    thresholds = detection_thresholds(cfg)
    model = YOLO(str(checkpoint), task="detect")
    predictions = _predictions_coco(model, dataset, thresholds)
    metrics = coco_metrics(dataset.coco_ground_truth(), predictions)
    return dict(
        metrics=metrics,
        predictions=predictions,
        evaluator=evaluator_metadata(cfg),
        split=split,
    )


def evaluate_split_native(cfg: dict[str, Any], checkpoint: Path, split: str, run_dir: Path) -> dict:
    from ultralytics import YOLO

    data_yaml = Path(cfg["data"]["root"]) / "data.yaml"
    model = YOLO(str(checkpoint), task="detect")
    results = model.val(
        data=str(data_yaml),
        split=split,
        device=_device_arg(cfg),
        imgsz=cfg["model"]["imgsz"],
        batch=cfg["eval"]["batch_size"],
        workers=cfg.get("runtime_cfg", {}).get("num_workers", 2),
        project=str(run_dir.resolve()),
        name=f"native_{split}",
        exist_ok=True,
    )
    return summarize_yolo(results)
