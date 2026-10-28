"""Rank-zero artifacts. No credentials or dataset image copies in run metadata."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

from .dataset import REPO_ROOT


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def provenance():
    import torch
    import torchvision
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            return None
    head, status, diff = git("rev-parse", "HEAD"), git("status", "--porcelain"), git("diff", "HEAD", "--", ".")
    # Include untracked source hashes; never store the diff text or environment values.
    untracked = git("ls-files", "--others", "--exclude-standard")
    extra = {}
    if untracked:
        from .dataset import file_hash
        for name in untracked.decode("utf-8").splitlines():
            p = REPO_ROOT/name
            if p.is_file() and p.suffix in {".py", ".yaml", ".toml", ".ipynb", ".md"}:
                extra[name] = file_hash(p)
    from .dataset import file_hash
    return dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        head=head.decode().strip() if head else None, dirty=bool(status) if status is not None else None,
        tracked_diff_sha256=hashlib.sha256(diff).hexdigest() if diff is not None else None,
        untracked_source_sha256=extra, python=platform.python_version(), torch=str(torch.__version__),
        torchvision=str(torchvision.__version__), cuda=torch.version.cuda,
        gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        manifest_sha256=file_hash(REPO_ROOT/"pyproject.toml"),
        local_pixi_lock_sha256=file_hash(REPO_ROOT/"pixi.lock") if (REPO_ROOT/"pixi.lock").exists() else None)


def curves(history, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, keys in (("loss", ["train_loss", "val_loss"]), ("lr", ["lr"]),
                       ("val_metrics", ["mAP50-95", "mAP50", "mAP75", "AR100"])):
        fig, ax = plt.subplots()
        for key in keys:
            ax.plot([r["epoch"] for r in history], [r.get(key) for r in history], label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory/f"{name}.png", dpi=140)
        plt.close(fig)
    # Composite results.png matching YOLO figures/results.png
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), tight_layout=True)
    for ax, (title, keys) in zip(
        axes,
        (
            ("loss", ["train_loss", "val_loss"]),
            ("lr", ["lr"]),
            ("val_metrics", ["mAP50-95", "mAP50", "mAP75", "AR100"]),
        ),
    ):
        for key in keys:
            ax.plot([r["epoch"] for r in history], [r.get(key) for r in history], marker=".", label=key)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.savefig(directory/"results.png", dpi=160)
    plt.close(fig)


def _box_iou_xyxy(boxes1, boxes2):
    import numpy as np
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0.0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def confusion_matrices(dataset, predictions, directory, *, conf_thresh=0.25, iou_thresh=0.45):
    """Compute and plot detection confusion_matrix.png and confusion_matrix_normalized.png
    using the same matching rule and layout as Ultralytics ConfusionMatrix.
    """
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = list(getattr(dataset, "names", [str(i) for i in range(52)]))
    nc = len(names)
    matrix = np.zeros((nc + 1, nc + 1), dtype=np.float64)

    preds_by_img: dict[int, list[dict]] = {}
    for p in predictions or []:
        if float(p.get("score", 0.0)) > conf_thresh:
            preds_by_img.setdefault(int(p["image_id"]), []).append(p)

    for record in getattr(dataset, "records", []):
        img_id = int(record["image_id"])
        gt_boxes_list = record.get("boxes", [])
        # ProcessedDataset stores 1-indexed labels (1..52); convert to 0-indexed (0..51)
        gt_cls = [int(lbl) - 1 for lbl in record.get("labels", [])]
        img_preds = preds_by_img.get(img_id, [])

        if not gt_cls:
            for p in img_preds:
                dc = int(p["category_id"])
                if 0 <= dc < nc:
                    matrix[dc, nc] += 1.0
            continue

        if not img_preds:
            for gc in gt_cls:
                if 0 <= gc < nc:
                    matrix[nc, gc] += 1.0
            continue

        gt_boxes = np.asarray(gt_boxes_list, dtype=np.float64).reshape(-1, 4)
        det_boxes = np.asarray(
            [[p["bbox"][0], p["bbox"][1], p["bbox"][0] + p["bbox"][2], p["bbox"][1] + p["bbox"][3]] for p in img_preds],
            dtype=np.float64,
        ).reshape(-1, 4)
        det_cls = [int(p["category_id"]) for p in img_preds]

        iou = _box_iou_xyxy(gt_boxes, det_boxes)
        gx, dx = np.where(iou > iou_thresh)
        if len(gx):
            matches = np.stack([gx, dx, iou[gx, dx]], axis=1)
            if len(matches) > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        matched_gt = set()
        matched_det = set()
        for m_gt, m_det, _ in matches:
            gi, di = int(m_gt), int(m_det)
            matched_gt.add(gi)
            matched_det.add(di)
            gc, dc = gt_cls[gi], det_cls[di]
            if 0 <= gc < nc and 0 <= dc < nc:
                matrix[dc, gc] += 1.0

        for gi, gc in enumerate(gt_cls):
            if gi not in matched_gt and 0 <= gc < nc:
                matrix[nc, gc] += 1.0
        for di, dc in enumerate(det_cls):
            if di not in matched_det and 0 <= dc < nc:
                matrix[dc, nc] += 1.0

    ticklabels = names + ["background"]
    nn = nc + 1
    xy_ticks = np.arange(nn)
    tick_fontsize = max(6, 15 - 0.1 * nn)
    label_fontsize = max(6, 12 - 0.1 * nn)
    title_fontsize = max(6, 12 - 0.1 * nn)
    btm = max(0.1, 0.25 - 0.001 * nn)

    for normalize, fname in ((False, "confusion_matrix.png"), (True, "confusion_matrix_normalized.png")):
        array = matrix / ((matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1.0)
        array = array.copy()
        array[array < 0.005] = np.nan
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = ax.imshow(array, cmap="Blues", vmin=0.0, interpolation="none")
            ax.xaxis.set_label_position("bottom")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
        title = "Confusion Matrix" + (" Normalized" if normalize else "")
        ax.set_xlabel("True", fontsize=label_fontsize, labelpad=10)
        ax.set_ylabel("Predicted", fontsize=label_fontsize, labelpad=10)
        ax.set_title(title, fontsize=title_fontsize, pad=20)
        ax.set_xticks(xy_ticks)
        ax.set_yticks(xy_ticks)
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        ax.tick_params(axis="y", left=True, right=False, labelleft=True, labelright=False)
        ax.set_xticklabels(ticklabels, fontsize=tick_fontsize, rotation=90, ha="center")
        ax.set_yticklabels(ticklabels, fontsize=tick_fontsize)
        for s in ("left", "right", "bottom", "top"):
            ax.spines[s].set_visible(False)
            cbar.ax.spines[s].set_visible(False)
        fig.subplots_adjust(left=0, right=0.84, top=0.94, bottom=btm)
        fig.savefig(directory / fname, dpi=200)
        plt.close(fig)


def example_ids(dataset, limit=12):
    positive = [r["image_id"] for r in dataset.records if r["labels"]]
    negative = [r["image_id"] for r in dataset.records if not r["labels"]]
    selected = negative[:limit//2]+positive[:limit//2]
    selected += [r["image_id"] for r in dataset.records if r["image_id"] not in selected][:limit-len(selected)]
    return selected


def overlays(dataset, predictions, ids, directory, checkpoint_hash):
    from PIL import Image, ImageDraw
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    selection = []
    for record in dataset.records:
        if record["image_id"] not in ids:
            continue
        with Image.open(record["path"]) as raw:
            image = raw.convert("RGB")
        draw = ImageDraw.Draw(image)
        for box, label in zip(record["boxes"], record["labels"]):
            draw.rectangle(box, outline="lime", width=2)
            draw.text((box[0], box[1]), f"GT {label-1}", fill="lime")
        for p in predictions:
            if p["image_id"] != record["image_id"]:
                continue
            x, y, w, h = p["bbox"]
            draw.rectangle([x, y, x+w, y+h], outline="red", width=1)
            draw.text((x, y+12), f"P {p['category_id']} {p['score']:.3f}", fill="red")
        image.save(directory/f"{record['image_id']:06d}.png")
        selection.append(dict(image_id=record["image_id"], file_name=record["path"].name))
    write_json(dict(selection_policy="fixed sorted positive/negative pool before inference; all selected images rendered",
        checkpoint_sha256=checkpoint_hash, examples=selection, gt_color="green", prediction_color="red"), directory/"examples.json")
