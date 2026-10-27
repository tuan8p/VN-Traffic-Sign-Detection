"""YOLO pipeline contract validation (no torch/ultralytics import)."""
from __future__ import annotations

import copy
import math
from pathlib import Path

WEIGHTS = "yolo26m.pt"
MODEL_DEFAULTS = dict(
    weights=WEIGHTS,
    imgsz=640,
    score_thresh=0.001,
    nms_thresh=0.5,
    detections_per_img=12,
)

TRAIN_DEFAULTS = dict(
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    lrf=0.01,
    warmup_epochs=3.0,
    cos_lr=True,
    close_mosaic=10,
)


def _number(value, name, low, high=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < low or (high is not None and value > high) or (integer and not isinstance(value, int)):
        raise ValueError(f"Invalid {name}: {value}")


def validate_config(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    if c.get("pipeline") != "yolo26m":
        raise ValueError("Expected pipeline=yolo26m")
    model = c.setdefault("model", {})
    for key, value in MODEL_DEFAULTS.items():
        model.setdefault(key, copy.deepcopy(value))
    train = c.setdefault("train", {})
    for key, value in TRAIN_DEFAULTS.items():
        train.setdefault(key, copy.deepcopy(value))
    allowed_model = set(MODEL_DEFAULTS) | {"family"}
    allowed_train = {
        "epochs", "lr0", "batch_size", "amp", "require_wandb",
        "early_stopping_patience", "run_name", "primary_metric",
        "momentum", "weight_decay", "lrf", "warmup_epochs", "cos_lr", "close_mosaic",
    }
    if set(model) - allowed_model:
        raise ValueError(f"Unknown model keys: {sorted(set(model) - allowed_model)}")
    if set(train) - allowed_train:
        raise ValueError(f"Unknown train keys: {sorted(set(train) - allowed_train)}")
    if model.get("family") != "yolo" or model["weights"] != WEIGHTS:
        raise ValueError("v1 supports YOLO26m publisher weights only")
    if model["imgsz"] != 640:
        raise ValueError("Processed cache and YOLO imgsz must be 640")
    if c.get("data", {}).get("num_classes") != 52:
        raise ValueError("Dataset must have 52 classes")
    for key in ("score_thresh", "nms_thresh"):
        _number(model[key], f"model.{key}", 0, 1)
    _number(model["detections_per_img"], "model.detections_per_img", 1, integer=True)
    for key in ("epochs", "batch_size", "early_stopping_patience"):
        _number(train[key], f"train.{key}", 1, integer=True)
    _number(c["eval"]["batch_size"], "eval.batch_size", 1, integer=True)
    if not isinstance(c["eval"].get("auto_eval_test", True), bool):
        raise ValueError("eval.auto_eval_test must be a boolean")
    if c["eval"].get("split", "test") != "test":
        raise ValueError("Final evaluation must use test split")
    if train.get("primary_metric", "mAP50-95") != "mAP50-95" or c.get("metrics", {}).get("primary") != "mAP50-95":
        raise ValueError("Best must be selected by validation mAP50-95")
    _number(train["lr0"], "train.lr0", 0)
    _number(train["momentum"], "train.momentum", 0, 1)
    _number(train["weight_decay"], "train.weight_decay", 0)
    _number(train["lrf"], "train.lrf", 0, 1)
    _number(train["warmup_epochs"], "train.warmup_epochs", 0)
    if not isinstance(train["cos_lr"], bool):
        raise ValueError("train.cos_lr must be a boolean")
    _number(train["close_mosaic"], "train.close_mosaic", 0, integer=True)
    _number(c.get("seed"), "seed", 0, 2**32 - 1, integer=True)
    online = c.setdefault("online_preprocess", {})
    if online.get("enabled", False) and online.get("hflip", False):
        raise ValueError("hflip is prohibited for mirror-pair sign classes")
    runtime = c.setdefault("runtime_cfg", {})
    if runtime.get("device", "cuda") not in ("cuda", "cpu"):
        raise ValueError("runtime_cfg.device must be cuda or cpu")
    root = c.get("data", {}).get("root")
    if not isinstance(root, (str, Path)) or not str(root).strip():
        raise ValueError("data.root is required")
    c["data"]["root"] = str(root)
    return c
