"""Validation of the Faster R-CNN contract, without importing ML libraries."""
from __future__ import annotations

import copy
import math
from pathlib import Path

WEIGHTS = "FasterRCNN_ResNet50_FPN_Weights.COCO_V1"
MODEL_DEFAULTS = dict(weights=WEIGHTS, trainable_backbone_layers=5, frozen_bn_eps=0.0,
                      score_thresh=0.001, nms_thresh=0.5, detections_per_img=12)
TRAIN_DEFAULTS = dict(scheduler="cosine", warmup_epochs=5, warmup_start_factor=0.001, min_lr=1e-6)
ONLINE_DEFAULTS = dict(mosaic_prob=1.0, mixup_prob=0.1, mixup_alpha=32.0,
                       color_jitter_params=dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02))
HEAD_KEYS = {"head", "head_hidden_dim", "head_dropout"}


def predictor_config(cfg):
    """Missing head fields mean the original standard predictor (legacy runs)."""
    m = cfg["model"]
    head = dict(name=m.get("head", "standard"))
    if head["name"] == "custom_mlp":
        head.update(version=1, hidden_dim=m.get("head_hidden_dim", 512), dropout=m.get("head_dropout", 0.2))
    return head


def _number(value, name, low, high=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < low or (high is not None and value > high) or (integer and not isinstance(value, int)):
        raise ValueError(f"Invalid {name}: {value}")


def validate_config(cfg):
    """Return a copy with explicit supported defaults; never mutate shared config."""
    c = copy.deepcopy(cfg)
    if c.get("pipeline") != "faster_rcnn":
        raise ValueError("Expected pipeline=faster_rcnn")
    for section, defaults in (("model", MODEL_DEFAULTS), ("train", TRAIN_DEFAULTS),
                              ("online_preprocess", ONLINE_DEFAULTS)):
        section_cfg = c.setdefault(section, {})
        for key, value in defaults.items():
            section_cfg.setdefault(key, copy.deepcopy(value))
    m, t, o = c["model"], c["train"], c["online_preprocess"]
    allowed_model = set(MODEL_DEFAULTS) | HEAD_KEYS | {"family", "backbone", "pretrained", "num_classes"}
    allowed_online = set(ONLINE_DEFAULTS) | {"enabled", "letterbox", "normalize", "hflip", "color_jitter", "mosaic", "mixup", "jitter"}
    allowed_train = set(TRAIN_DEFAULTS) | {"epochs", "lr", "momentum", "weight_decay", "primary_metric",
        "batch_size", "amp", "require_wandb", "early_stopping_patience", "run_name"}
    for name, actual, allowed in (("model", m, allowed_model), ("train", t, allowed_train), ("online_preprocess", o, allowed_online)):
        if set(actual) - allowed:
            raise ValueError(f"Unknown {name} keys: {sorted(set(actual)-allowed)}")
    if m.get("family") != "faster_rcnn" or m.get("backbone") != "resnet50" or m["weights"] != WEIGHTS:
        raise ValueError("v1 supports Faster R-CNN ResNet50 FPN COCO_V1 only")
    if m.get("num_classes") != 52 or c.get("data", {}).get("num_classes") != 52:
        raise ValueError("Dataset must have 52 classes; predictor has 53 outputs")
    if m["trainable_backbone_layers"] != 5 or m["frozen_bn_eps"] != 0:
        raise ValueError("v1 requires full finetuning and FrozenBN eps=0")
    if m.get("head", "standard") not in ("standard", "custom_mlp"):
        raise ValueError("model.head must be standard or custom_mlp")
    _number(m.get("head_hidden_dim", 512), "model.head_hidden_dim", 1, integer=True)
    _number(m.get("head_dropout", 0.2), "model.head_dropout", 0, 1)
    if m.get("head_dropout", 0.2) == 1:
        raise ValueError("model.head_dropout must be <1")
    if c.get("preprocess", {}).get("image_size") != [640, 640]:
        raise ValueError("Processed cache must be 640x640")
    if c["preprocess"].get("pad_value") != 114 or c["preprocess"].get("hflip", False):
        raise ValueError("Expected pad114 and no offline hflip")
    for section, keys in ((m, ["pretrained"]), (t, ["amp", "require_wandb"]),
                          (o, ["enabled", "letterbox", "normalize", "hflip", "color_jitter", "mosaic", "mixup"])):
        for key in keys:
            if key in section and not isinstance(section[key], bool):
                raise ValueError(f"{key} must be a boolean")
    for k in ("score_thresh", "nms_thresh"):
        _number(m[k], f"model.{k}", 0, 1)
    _number(m["detections_per_img"], "model.detections_per_img", 1, integer=True)
    for k in ("epochs", "batch_size", "early_stopping_patience"):
        _number(t[k], f"train.{k}", 1, integer=True)
    _number(c["eval"]["batch_size"], "eval.batch_size", 1, integer=True)
    if not isinstance(c["eval"].get("auto_eval_test", True), bool):
        raise ValueError("eval.auto_eval_test must be a boolean")
    if c["eval"].get("split", "test") != "test":
        raise ValueError("Standalone/final evaluation must use test; epoch selection uses val internally")
    _number(c["preprocess"].get("min_box_px"), "min_box_px", 0)
    _number(c.get("seed"), "seed", 0, 2**32-1, integer=True)
    for k in ("lr", "weight_decay", "momentum"):
        _number(t[k], f"train.{k}", 0)
    if t["lr"] <= 0 or t["momentum"] >= 1:
        raise ValueError("Require positive LR and momentum <1")
    if t["scheduler"] != "cosine":
        raise ValueError("v1 supports cosine scheduler only")
    _number(t["warmup_epochs"], "train.warmup_epochs", 1, integer=True)
    _number(t["warmup_start_factor"], "warmup_start_factor", 0, 1)
    _number(t["min_lr"], "min_lr", 0, t["lr"])
    if t["warmup_start_factor"] == 0:
        raise ValueError("warmup_start_factor must be positive")
    if t.get("primary_metric", "mAP50-95") != "mAP50-95" or c.get("metrics", {}).get("primary") != "mAP50-95":
        raise ValueError("Best must be selected by validation mAP50-95")
    if o.get("enabled", False):
        if o.get("hflip", False):
            raise ValueError("hflip is prohibited for mirror-pair sign classes")
        _number(o.get("jitter", 0), "jitter", 0)
        if o.get("jitter", 0) >= 1:
            raise ValueError("jitter must be <1")
        for k in ("mosaic_prob", "mixup_prob"):
            _number(o[k], k, 0, 1)
        _number(o["mixup_alpha"], "mixup_alpha", 0)
        if o["mixup_alpha"] == 0:
            raise ValueError("mixup_alpha must be positive")
        params = o["color_jitter_params"]
        if set(params) != {"brightness", "contrast", "saturation", "hue"}:
            raise ValueError("Expected brightness, contrast, saturation, hue")
        for k, v in params.items():
            _number(v, k, 0, 0.5 if k == "hue" else None)
    runtime = c.setdefault("runtime_cfg", {})
    if runtime.get("device", "cuda") not in ("cuda", "cpu"):
        raise ValueError("runtime_cfg.device must be cuda or cpu")
    _number(runtime.get("num_workers", 0), "num_workers", 0, integer=True)
    for k in ("mixed_precision", "pin_memory"):
        if k in runtime and not isinstance(runtime[k], bool):
            raise ValueError(f"runtime_cfg.{k} must be boolean")
    root = c.get("data", {}).get("root")
    if not isinstance(root, (str, Path)) or not str(root).strip():
        raise ValueError("data.root is required; no implicit cache fallback")
    c["data"]["root"] = str(root)
    if c["data"].get("images_dirname", "images") != "images" or c["data"].get("labels_dirname", "labels") != "labels":
        raise ValueError("Processed contract requires images/ and labels/ directories")
    # Never serialize credentials supplied accidentally in config/overrides.
    def check_secret(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() in {"api_key", "wandb_api_key", "password", "token", "secret"}:
                    raise ValueError("Credentials belong in environment/Secrets, not configuration")
                check_secret(v)
        elif isinstance(obj, list):
            for v in obj:
                check_secret(v)
    check_secret(c)
    return c
