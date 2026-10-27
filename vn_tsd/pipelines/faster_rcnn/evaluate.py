"""COCO bbox metrics and state-preserving validation loss."""
from __future__ import annotations

import contextlib
import random

import numpy as np
import torch

from vn_tsd.eval.coco_eval import coco_metrics, evaluator_metadata

LOSS_KEYS = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")


def to_device(images, targets, device):
    return [im.to(device) for im in images], [{k: v.to(device) for k, v in t.items()} for t in targets]


def infer(model, loader, device):
    model.eval()
    predictions, seen = [], []
    with torch.inference_mode():
        for images, targets in loader:
            outputs = model([im.to(device) for im in images])
            if len(outputs) != len(targets):
                raise ValueError("Model output/image count mismatch")
            for output, target in zip(outputs, targets):
                image_id = int(target["image_id"])
                seen.append(image_id)
                boxes, scores, labels = (output[k].detach().cpu() for k in ("boxes", "scores", "labels"))
                if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
                    raise ValueError("Nonfinite model prediction")
                if len(boxes) != len(scores) or len(boxes) != len(labels):
                    raise ValueError("Invalid prediction shapes")
                for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                    x1, y1, x2, y2 = box
                    if not 1 <= label <= 52 or not 0 <= score <= 1 or x2 <= x1 or y2 <= y1:
                        raise ValueError("Invalid prediction category/score/box")
                    predictions.append(dict(image_id=image_id, category_id=label-1,
                        bbox=[x1, y1, x2-x1, y2-y1], score=score))
    return predictions, seen


def distributed_evaluate(model, dataset, loader, ctx):
    ctx.sync_buffers(model)
    local = ctx.phase(lambda: infer(model, loader, ctx.device))
    gathered = ctx.gather(local)
    def finish():
        predictions = [p for rows, _ in gathered for p in rows]
        ids = [i for _, seen in gathered for i in seen]
        expected = [r["image_id"] for r in dataset.records]
        if sorted(ids) != sorted(expected):
            raise ValueError("Evaluation coverage must contain every image exactly once")
        return coco_metrics(dataset.coco_ground_truth(), predictions), predictions
    return ctx.main_call(finish)


@contextlib.contextmanager
def diagnostic_mode(model, seed):
    """RPN/RoI loss needs train mode; preserve flags, RNG and mutable buffers."""
    flags = [(m, m.training) for m in model.modules()]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    py_state, np_state = random.getstate(), np.random.get_state()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
            model.train()
            for m in model.modules():
                if isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd)):
                    m.eval()
            yield
    finally:
        for b, saved in buffers:
            b.copy_(saved)
        for m, training in flags:
            m.training = training
        random.setstate(py_state)
        np.random.set_state(np_state)


def validation_loss(model, loader, device, seed):
    sums = {k: 0.0 for k in LOSS_KEYS}
    count = 0
    with torch.no_grad(), diagnostic_mode(model, seed):
        for images, targets in loader:
            images, targets = to_device(images, targets, device)
            loss = model(images, targets)
            if set(loss) != set(LOSS_KEYS):
                raise ValueError("Expected four Faster R-CNN loss components")
            for k in LOSS_KEYS:
                value = float(loss[k])
                if not np.isfinite(value):
                    raise ValueError("Nonfinite validation loss")
                sums[k] += value*len(images)
            count += len(images)
    return [sums[k] for k in LOSS_KEYS]+[count]
