"""Detection metrics. Primary: mAP@0.50:0.95. Also mAP@0.50, mAP@0.75, AR@100.

Same names as configs/shared.yaml metrics: mAP50-95, mAP50, mAP75, AR100.
TODO [Team pipeline YOLO26m + preprocessing online]: Ultralytics.
TODO [Team pipeline Faster R-CNN + preprocessing online]: torchvision.
"""
from __future__ import annotations
from typing import Any

def summarize(raw: dict[str, Any]) -> dict[str, float]:
    # TODO [Team pipeline YOLO26m + preprocessing online]: map Ultralytics val outputs onto these four keys.
    # TODO [Team pipeline Faster R-CNN + preprocessing online]: map torchvision / COCO eval outputs onto the same schema.
    return {
        "mAP50-95": float(raw.get("mAP50-95", 0.0)),
        "mAP50": float(raw.get("mAP50", 0.0)),
        "mAP75": float(raw.get("mAP75", 0.0)),
        "AR100": float(raw.get("AR100", 0.0)),
    }
