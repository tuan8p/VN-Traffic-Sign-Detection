"""Faster R-CNN configuration checks, independent of model/training dependencies."""
from pathlib import Path

import pytest

from vn_tsd.config.resolve import resolve_config
from vn_tsd.pipelines.faster_rcnn.config import validate_config

ROOT = Path(__file__).resolve().parents[1]


def resolved_config():
    return resolve_config(
        ROOT / "configs/pipelines/faster_rcnn.yaml",
        ROOT / "configs/shared.yaml",
        ROOT / "configs/runtime/local.yaml",
        dict(data=dict(root="fixture")),
    )


@pytest.mark.parametrize("section,key,value", [
    ("model", "backbone", "resnet18"), ("model", "num_classes", 53),
    ("model", "trainable_backbone_layers", 3), ("model", "frozen_bn_eps", 1e-5),
    ("train", "warmup_epochs", 2), ("train", "warmup_start_factor", 0),
    ("train", "lr", float("nan")), ("train", "epohcs", 2),
    ("runtime_cfg", "device", "auto"), ("online_preprocess", "enabled", "false"),
    ("eval", "auto_eval_test", "false"), ("data", "root", None), ("train", "batch_size", 8.0),
])
def test_invalid_config_is_not_silently_ignored(section, key, value):
    cfg = resolved_config()
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_validation_accepts_baseline_without_mutating_resolved_config():
    import copy

    cfg = resolved_config()
    before = copy.deepcopy(cfg)
    validated = validate_config(cfg)
    assert validated == before
    assert cfg == before
    assert validated is not cfg
