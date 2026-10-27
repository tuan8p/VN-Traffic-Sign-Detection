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
    ("model", "head", "unknown"), ("model", "head_hidden_dim", 0),
    ("model", "head_hidden_dim", 12.5), ("model", "head_dropout", 1),
    ("model", "head_dropout", float("nan")),
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


@pytest.mark.parametrize("head", ["standard", "custom_mlp"])
def test_head_selection_is_explicit(head):
    from vn_tsd.pipelines.faster_rcnn.config import predictor_config

    cfg = resolved_config()
    cfg["model"]["head"] = head
    result = predictor_config(validate_config(cfg))
    assert result["name"] == head
    if head == "custom_mlp":
        assert result["hidden_dim"] == 512 and result["dropout"] == 0.2


def test_legacy_config_does_not_gain_new_head_fields():
    from vn_tsd.pipelines.faster_rcnn.config import predictor_config

    cfg = resolved_config()
    for key in ("head", "head_hidden_dim", "head_dropout"):
        cfg["model"].pop(key)
    validated = validate_config(cfg)
    assert validated == cfg
    assert predictor_config(validated) == {"name": "standard"}
