"""Writable YOLO dataset tree for Ultralytics (.cache) on read-only Kaggle Input."""
from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path

from vn_tsd.utils.io import load_yaml, save_yaml

KAGGLE_INPUT = Path("/kaggle/input")


def _on_kaggle_input(source_root: Path) -> bool:
    if not KAGGLE_INPUT.is_dir():
        return str(source_root).startswith("/kaggle/input")
    try:
        source_root.resolve().relative_to(KAGGLE_INPUT.resolve())
        return True
    except ValueError:
        return str(source_root.resolve()).startswith("/kaggle/input")


def _ready(root: Path) -> bool:
    return (root / "data.yaml").is_file() and (root / "images" / "train").exists()


def _link_dir(src: Path, dst: Path, *, allow_copy_fallback: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError as exc:
        if not allow_copy_fallback:
            raise RuntimeError(
                f"Symlink failed ({src} -> {dst}). On Kaggle use YOLO_SKIP_DATASET_STAGE=True "
                f"or fix permissions; do not copy thousands of files from Input. Original: {exc}"
            ) from exc
        shutil.copytree(src, dst)


def prepare_yolo_writable_dataset(source_root: Path, working: Path) -> Path:
    """Return a dataset root Ultralytics can write label ``.cache`` files under."""
    source_root = source_root.resolve()
    working = working.resolve()
    kaggle_input = _on_kaggle_input(source_root)

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
    allow_copy = not kaggle_input

    if kaggle_input:
        _stage_kaggle_symlinks(source_root, fixed_root)
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


def _stage_kaggle_symlinks(source_root: Path, fixed_root: Path) -> None:
    """Symlink only — copying from Input can take 10–30+ minutes."""
    print("[yolo dataset] Kaggle Input: symlinks only (vài giây)…", flush=True)
    for name in ("data.yaml", "meta.json"):
        src = source_root / name
        if src.is_file():
            shutil.copy2(src, fixed_root / name)
    for split in ("train", "val", "test"):
        print(f"[yolo dataset] link {split}…", flush=True)
        _link_dir(
            source_root / "images" / split,
            fixed_root / "images" / split,
            allow_copy_fallback=False,
        )
        _link_dir(
            source_root / "labels" / split,
            fixed_root / "labels" / split,
            allow_copy_fallback=False,
        )
    print("[yolo dataset] done.", flush=True)
