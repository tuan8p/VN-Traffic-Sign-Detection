"""Shared detection metric schema. Values are fractions; unavailable means None.

Adapters supply metrics from their evaluator. Normalizing names does not make
Ultralytics AP numerically identical to COCOeval AP.
"""
from __future__ import annotations

import math
from typing import Any

METRIC_KEYS = ("mAP50-95", "mAP50", "mAP75", "AR100")
PRIMARY_METRIC = "mAP50-95"

# Contract dùng chung: bốn khóa trên, giá trị [0, 1]; thiếu dùng None, không điền 0.
# TODO [Team pipeline Faster R-CNN + preprocessing online]: ánh xạ COCOeval bbox
# Team Faster R-CNN maps COCOeval bbox in pipelines/faster_rcnn/evaluate.py:
# stats[0], [1], [2], [8] lần lượt vào AP, AP50, AP75, AR100, với cấu hình
# maxDets=[1, 10, 100], area=all. Đổi giá trị không xác định (-1) thành None,
# rồi gọi summarize(raw); không truyền trực tiếp labels có background vào COCO.
# TODO [Hai team pipeline]: dùng chung COCO evaluator cho báo cáo cuối cùng;
# schema giống nhau chưa bảo đảm AP native Ultralytics bằng AP của COCOeval.
# Xem docs/METRICS_CONTRACT.md để thống nhất class mapping và split.
# Evaluator parity is tracked in docs/FRCNN_TEAM_HANDOFF.md.


def summarize(raw: dict[str, Any]) -> dict[str, float | None]:
    """Validate available metrics without fabricating scores for missing ones."""
    result: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        value = raw.get(key)
        if value is None:
            result[key] = None
            continue
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a fraction in [0, 1], not a boolean")
        score = float(value)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"{key} must be a finite fraction in [0, 1]: {value}")
        result[key] = score
    return result


def summarize_yolo(results: Any) -> dict[str, float | None]:
    """Map native Ultralytics AP; native mean recall is NOT COCO AR@100."""
    # TODO [Team pipeline YOLO26m + preprocessing online]: tính COCO AR@100
    # từ predictions/ground truth; không dùng results.box.mr thay thế.
    return summarize({
        "mAP50-95": results.box.map,
        "mAP50": results.box.map50,
        "mAP75": results.box.map75,
        "AR100": None,
    })
