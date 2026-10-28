"""Shared COCO bbox evaluation for both detection pipelines."""
from __future__ import annotations

import contextlib
import copy
import importlib.metadata
import io
from typing import Any

from vn_tsd.eval.metrics import summarize

DEFAULT_THRESHOLDS = dict(score_thresh=0.001, nms_thresh=0.5, detections_per_img=100)


def detection_thresholds(cfg: dict[str, Any]) -> dict[str, float | int]:
    model = cfg.get("model", {})
    return dict(
        score_thresh=float(model.get("score_thresh", DEFAULT_THRESHOLDS["score_thresh"])),
        nms_thresh=float(model.get("nms_thresh", DEFAULT_THRESHOLDS["nms_thresh"])),
        detections_per_img=int(model.get("detections_per_img", DEFAULT_THRESHOLDS["detections_per_img"])),
    )


def coco_metrics(ground_truth: dict, predictions: list[dict]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = copy.deepcopy(ground_truth)
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = dict(
                images=gt.dataset["images"],
                categories=gt.dataset["categories"],
                annotations=[],
            )
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    values = [None if evaluator.stats[i] == -1 else float(evaluator.stats[i]) for i in (0, 1, 2, 8)]
    return summarize(dict(zip(("mAP50-95", "mAP50", "mAP75", "AR100"), values)))


def evaluator_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    thresholds = detection_thresholds(cfg)
    try:
        version = importlib.metadata.version("pycocotools")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return dict(
        evaluator="pycocotools.COCOeval bbox",
        version=version,
        iou="0.50:0.05:0.95",
        area="all",
        maxDets=[1, 10, 100],
        stats_indices=[0, 1, 2, 8],
        **thresholds,
    )
