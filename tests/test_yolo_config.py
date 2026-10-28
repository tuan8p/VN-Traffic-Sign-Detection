from pathlib import Path

import pytest

from vn_tsd.config.resolve import resolve_config
from vn_tsd.pipelines.yolo.config import validate_config

ROOT = Path(__file__).resolve().parents[1]


def resolved():
    return resolve_config(
        ROOT / "configs/pipelines/yolo26m.yaml",
        ROOT / "configs/shared.yaml",
        ROOT / "configs/runtime/local.yaml",
        dict(data=dict(root="fixture")),
    )


@pytest.mark.parametrize("section,key,value", [
    ("metrics", "primary", "mAP50"),
    ("train", "primary_metric", "mAP50"),
    ("train", "optimizer", "SGD"),
    ("train", "optimizer", "auto"),
    ("model", "weights", "yolo11n.pt"),
    ("model", "imgsz", 512),
])
def test_invalid_yolo_config(section, key, value):
    cfg = resolved()
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_yolo_config_sets_coco_threshold_defaults():
    cfg = resolved()
    validated = validate_config(cfg)
    assert validated["model"]["score_thresh"] == 0.001
    assert validated["model"]["detections_per_img"] == 100
    # Explicit AdamW (not Ultralytics auto, which ignores lr0/momentum).
    assert validated["train"]["optimizer"] == "AdamW"
    assert validated["train"]["lr0"] == 0.0002
    assert validated["train"]["momentum"] == 0.9
    assert validated["online_preprocess"]["enabled"] is True
