"""Make a writable YOLO dataset tree for Ultralytics label caches on read-only Kaggle Input."""
from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path

from vn_tsd.utils.io import load_yaml, save_yaml


def _ready(root: Path) -> bool:
    return (root / "data.yaml").is_file() and (root / "images" / "train").exists()


def prepare_yolo_writable_dataset(source_root: Path, working: Path) -> Path:
    """Return a path Ultralytics can write label ``.cache`` files under."""
    source_root = source_root.resolve()
    working = working.resolve()

    if str(source_root).startswith(str(working)) and _ready(source_root):
        return source_root

    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            if not (source_root / kind / split).is_dir():
                raise FileNotFoundError(source_root / kind / split)

    meta = source_root / "meta.json"
    if meta.is_file():
        tag = hashlib.sha256(meta.read_bytes()).hexdigest()[:8]
    else:
        tag = uuid.uuid4().hex[:8]
    fixed_root = working / f"yolo_dataset_{tag}"

    if _ready(fixed_root):
        _patch_data_yaml(fixed_root)
        return fixed_root

    fixed_root.mkdir(parents=True, exist_ok=True)

    if str(source_root).startswith("/kaggle/input"):
        _stage_kaggle_input(source_root, fixed_root)
    else:
        if fixed_root.exists():
            shutil.rmtree(fixed_root)
        shutil.copytree(source_root, fixed_root, ignore=shutil.ignore_patterns("*.cache"))

    _patch_data_yaml(fixed_root)
    return fixed_root


def _patch_data_yaml(root: Path) -> None:
    data_cfg = load_yaml(root / "data.yaml")
    data_cfg.update(
        path=str(root),
        train="images/train",
        val="images/val",
        test="images/test",
    )
    save_yaml(data_cfg, root / "data.yaml")


def _stage_kaggle_input(source_root: Path, fixed_root: Path) -> None:
    """Copy labels + metadata; symlink image splits (read-only Input is slow to duplicate)."""
    fixed_root.mkdir(parents=True, exist_ok=True)
    for name in ("data.yaml", "meta.json"):
        src = source_root / name
        if src.is_file():
            shutil.copy2(src, fixed_root / name)
    for split in ("train", "val", "test"):
        src_img = source_root / "images" / split
        dst_img = fixed_root / "images" / split
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        try:
            dst_img.symlink_to(src_img, target_is_directory=True)
        except OSError:
            shutil.copytree(src_img, dst_img)
        src_lbl = source_root / "labels" / split
        dst_lbl = fixed_root / "labels" / split
        if dst_lbl.exists():
            shutil.rmtree(dst_lbl)
        shutil.copytree(src_lbl, dst_lbl)
