"""ResNet50-FPN COCO_V1 and offline checkpoint reconstruction.

Reference: torchvision.models.detection.faster_rcnn and backbone_utils.
FrozenBN epsilon is explicit because state_dict does not serialize it.
"""
from __future__ import annotations

from .config import predictor_config

SCHEMA = "vn-tsd-frcnn-inference-v1"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def architecture(cfg):
    m = cfg["model"]
    result = dict(family="faster_rcnn", backbone="resnet50_fpn", num_classes=53,
        weights=m["weights"], trainable_backbone_layers=5, norm="FrozenBatchNorm2d", frozen_bn_eps=0.0,
        min_size=640, max_size=640, size_divisible=32, mean=MEAN, std=STD,
        score_thresh=m["score_thresh"], nms_thresh=m["nms_thresh"], detections_per_img=100,
        input="RGB float32 CHW [0,1]", coordinates="processed image pixels xyxy")
    # Legacy checkpoints without a head field retain their exact metadata/hash.
    if "head" in m:
        result["predictor"] = predictor_config(cfg)
    return result


def build_faster_rcnn(cfg, *, initialize=None):
    """initialize=False guarantees no weight download, including backbone weights."""
    from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.ops.misc import FrozenBatchNorm2d
    a = architecture(cfg)
    initialize = cfg["model"].get("pretrained", True) if initialize is None else initialize
    backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=None,
                                   norm_layer=FrozenBatchNorm2d, trainable_layers=5)
    model = FasterRCNN(backbone, num_classes=91 if initialize else 53,
        min_size=a["min_size"], max_size=a["max_size"], image_mean=a["mean"], image_std=a["std"],
        box_score_thresh=a["score_thresh"], box_nms_thresh=a["nms_thresh"], box_detections_per_img=100)
    if initialize:
        state = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.get_state_dict(progress=True, check_hash=True)
        model.load_state_dict(state, strict=True)
    # Load the entire COCO state strictly BEFORE installing either task head.
    # This preserves pretrained backbone/FPN/RPN/TwoMLPHead on both variants.
    head = predictor_config(cfg)
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    if head["name"] == "custom_mlp":
        from .heads import CustomFastRCNNPredictor
        model.roi_heads.box_predictor = CustomFastRCNNPredictor(
            in_channels, 53, hidden_dim=head["hidden_dim"], dropout=head["dropout"])
    elif head["name"] == "standard":
        if initialize:
            model.roi_heads.box_predictor = FastRCNNPredictor(in_channels, 53)
    else:
        raise ValueError(f"Unsupported predictor: {head['name']}")
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = 0.0
    return model


def save_checkpoint(path, model, metadata):
    import os
    from pathlib import Path
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    try:
        torch.save(dict(schema=SCHEMA, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        metadata=metadata), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_checkpoint(path):
    import torch
    from .config import validate_config
    from .dataset import content_hash
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or obj.get("schema") != SCHEMA:
        raise ValueError("Unsupported Faster R-CNN inference checkpoint schema")
    meta = obj["metadata"]
    cfg = validate_config(meta["resolved_config"])
    if meta["architecture"] != architecture(cfg):
        raise ValueError("Checkpoint architecture/normalization/epsilon mismatch")
    if content_hash(cfg) != meta["config_sha256"]:
        raise ValueError("Checkpoint resolved configuration hash mismatch")
    names = meta["dataset"]["names"]
    if len(names) != 52 or content_hash(names) != meta["dataset"]["class_map_sha256"]:
        raise ValueError("Checkpoint class mapping mismatch")
    return obj


def load_checkpoint(path, device="cpu"):
    """Return (eval model, metadata), without downloading or starting W&B."""
    obj = read_checkpoint(path)
    model = build_faster_rcnn(obj["metadata"]["resolved_config"], initialize=False)
    model.load_state_dict(obj["state_dict"], strict=True)
    model.to(device).eval()
    return model, obj["metadata"]
