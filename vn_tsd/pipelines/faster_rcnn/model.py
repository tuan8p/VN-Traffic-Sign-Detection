"""Faster R-CNN with a ResNet-50 FPN backbone (torchvision).

TODO [Team pipeline Faster R-CNN + preprocessing online]: build torchvision.models.detection.fasterrcnn_resnet50_fpn,
replace the box predictor head for num_classes + background.
"""
from __future__ import annotations
from typing import Any

def build_faster_rcnn(cfg: dict[str, Any]):
    raise NotImplementedError(
        "TODO [Team pipeline Faster R-CNN + preprocessing online]: torchvision fasterrcnn_resnet50_fpn, dataset has 52 classes; box predictor must be 53 including background"
    )
