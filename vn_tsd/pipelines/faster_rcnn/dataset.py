"""Strict reader of the team's processed YOLO cache. Never rewrites the cache."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from vn_tsd.utils.io import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def content_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def read_class_names(path):
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 52 or [int(r["class_id"]) for r in rows] != list(range(52)):
        raise ValueError("Class table must contain ordered IDs 0..51")
    return [r["name_vi"] for r in rows], file_hash(path)


def parse_labels(path, size=640):
    boxes, labels = [], []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            if len(parts) != 5 or not parts[0].isdigit():
                raise ValueError("expected integer class and four coordinates")
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            if not 0 <= cls < 52 or not all(math.isfinite(v) for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
                raise ValueError("invalid class or box")
            xyxy = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
            if min(xyxy) < -1e-5 or max(xyxy) > 1+1e-5:
                raise ValueError("box outside image beyond rounding tolerance")
            xyxy = [max(0.0, min(1.0, v))*size for v in xyxy]
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                raise ValueError("degenerate box after clipping")
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        boxes.append(xyxy)
        labels.append(cls+1)
    return boxes, labels


class ProcessedDataset:
    """Map dataset with stable IDs; training augmentation is injected, not cached."""

    def __init__(self, root, split, names, transform=None, min_box_px=8):
        from PIL import Image
        self.root, self.split, self.names = Path(root).expanduser().resolve(), split, list(names)
        self.transform = transform
        image_dir, label_dir = self.root/"images"/split, self.root/"labels"/split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing processed split: {image_dir}, {label_dir}")
        if any(p.is_dir() for p in image_dir.iterdir()) or any(p.is_dir() for p in label_dir.iterdir()):
            raise ValueError("Processed split directories must be flat")
        images = sorted((p for p in image_dir.iterdir() if p.suffix.lower() in EXTENSIONS), key=lambda p: p.name)
        if not images:
            raise ValueError(f"Empty split {split}")
        stems = [p.stem for p in images]
        if len(stems) != len(set(stems)):
            raise ValueError(f"Duplicate image stem in {split}")
        label_files = [p for p in label_dir.iterdir() if p.suffix.lower() == ".txt"]
        labels = {p.stem: p for p in label_files}
        if len(labels) != len(label_files):
            raise ValueError(f"Duplicate label stem in {split}")
        if set(stems) != set(labels):
            raise ValueError(f"Missing/orphan labels in {split}: {sorted(set(stems)^set(labels))[:10]}")
        self.records, self.manifest, self.small_boxes = [], [], 0
        for i, image in enumerate(images):
            with Image.open(image) as im:
                if im.size != (640, 640):
                    raise ValueError(f"Expected 640x640 cache: {image} has {im.size}")
                im.convert("RGB").load()  # decode now so corrupted images fail preflight
            boxes, targets = parse_labels(labels[image.stem])
            self.small_boxes += sum(min(b[2]-b[0], b[3]-b[1]) < min_box_px-1e-3 for b in boxes)
            self.records.append(dict(path=image, boxes=boxes, labels=targets, image_id=i+1))
            self.manifest.append(dict(image=str(image.relative_to(self.root)).replace("\\", "/"),
                label=str(labels[image.stem].relative_to(self.root)).replace("\\", "/"),
                image_sha256=file_hash(image), label_sha256=file_hash(labels[image.stem])))
        self.identity = dict(count=len(self.records), annotations=sum(len(r["labels"]) for r in self.records),
                             manifest=self.manifest, sha256=content_hash(self.manifest), small_boxes=self.small_boxes)

    def __len__(self):
        return len(self.records)

    def raw(self, index):
        import torch
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor
        r = self.records[index]
        with Image.open(r["path"]) as im:
            image = pil_to_tensor(im.convert("RGB")).float().div_(255)
        return image, {"boxes": torch.tensor(r["boxes"], dtype=torch.float32).reshape(-1, 4),
                       "labels": torch.tensor(r["labels"], dtype=torch.int64),
                       "image_id": torch.tensor(r["image_id"], dtype=torch.int64)}

    def __getitem__(self, index):
        import torch
        image, target = self.raw(index) if self.transform is None else self.transform(self, index)
        b = target["boxes"]
        target["area"] = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
        target["iscrowd"] = torch.zeros(len(b), dtype=torch.int64)
        return image, target

    def coco_ground_truth(self):
        images, annotations = [], []
        for r in self.records:
            images.append(dict(id=r["image_id"], width=640, height=640, file_name=r["path"].name))
            for box, label in zip(r["boxes"], r["labels"]):
                x1, y1, x2, y2 = box
                annotations.append(dict(id=len(annotations)+1, image_id=r["image_id"], category_id=label-1,
                    bbox=[x1, y1, x2-x1, y2-y1], area=(x2-x1)*(y2-y1), iscrowd=0))
        return dict(info={}, images=images, annotations=annotations,
                    categories=[dict(id=i, name=n) for i, n in enumerate(self.names)])


def load_cache(cfg, splits=("train", "val", "test"), saved_names=None):
    root = Path(cfg["data"]["root"]).expanduser().resolve()
    data = load_yaml(root/"data.yaml")
    meta_path = root/"meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("meta.json must be an object")
    if saved_names is None:
        names, table_hash = read_class_names(cfg["data"]["class_table"])
    else:
        names, table_hash = list(saved_names), None
    yaml_names = data.get("names")
    if isinstance(yaml_names, dict):
        try:
            indexed = {int(k): v for k, v in yaml_names.items()}
            if len(indexed) != len(yaml_names) or set(indexed) != set(range(52)):
                raise ValueError("Expected class IDs 0..51")
            yaml_names = [indexed[i] for i in range(52)]
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid data.yaml names") from exc
    if data.get("nc") != 52 or len(names) != 52 or yaml_names != names:
        raise ValueError("data.yaml nc/names do not match ordered class map")
    sets = {s: ProcessedDataset(root, s, names, min_box_px=cfg["preprocess"]["min_box_px"]) for s in splits}
    hashes = {}
    for split, dataset in sets.items():
        for r in dataset.manifest:
            digest = r["image_sha256"]
            if digest in hashes and hashes[digest] != split:
                raise ValueError(f"Byte-identical image across splits: {hashes[digest]} and {split}")
            hashes[digest] = split
    identity = dict(root=str(root), names=names, class_map_sha256=content_hash(names),
        class_table_sha256=table_hash, data_yaml_sha256=file_hash(root/"data.yaml"), meta_sha256=file_hash(meta_path),
        splits={s: d.identity for s, d in sets.items()}, byte_hash_check="does not detect re-encoded/near duplicates")
    return sets, identity


def collate_detection(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
