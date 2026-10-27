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
