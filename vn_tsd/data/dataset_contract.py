"""Processed cache contract: train/val/test must live under data.root.

The only allowed external assets are pretrain checkpoints (FRCNN COCO weights,
YOLO yolo26m.pt). No external val/test sets or alternate data URLs.
"""
from __future__ import annotations

from pathlib import Path

from vn_tsd.utils.io import load_yaml, save_yaml

SPLITS = ("train", "val", "test")


def _resolve_data_yaml_path_field(raw_path: str, data_yaml: Path) -> Path | None:
    """Resolve Ultralytics ``path``; return None for stale foreign (e.g. Windows) paths on POSIX."""
    text = str(raw_path).strip()
    if not text:
        return None
    if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
        return None
    declared = Path(text).expanduser()
    if not declared.is_absolute():
        declared = (data_yaml.parent / declared).resolve()
    else:
        declared = declared.resolve()
    return declared


def validate_processed_dataset(cfg: dict) -> Path:
    """Return resolved data root after validating layout and split paths."""
    root_raw = cfg.get("data", {}).get("root")
    if not isinstance(root_raw, str) or not root_raw.strip():
        raise ValueError("data.root is required; training/eval use only the processed cache")
    root = Path(root_raw).expanduser().resolve()
    data_yaml = root / "data.yaml"
    meta_json = root / "meta.json"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Missing dataset config under data.root: {data_yaml}")
    if not meta_json.is_file():
        raise FileNotFoundError(f"Missing preprocess meta under data.root: {meta_json}")

    data = load_yaml(data_yaml)
    yaml_path = data.get("path")
    if yaml_path is not None:
        declared = _resolve_data_yaml_path_field(str(yaml_path), data_yaml)
        if declared is not None and declared != root:
            # data.root (--data-root) is authoritative; path is often an absolute path from the build machine.
            print(
                f"[dataset] data.yaml path ({yaml_path!r}) != data.root ({root}); using data.root",
                flush=True,
            )

    for key in ("nc", "names"):
        if key not in data:
            raise ValueError(f"data.yaml must define {key} for the processed export")

    for split in SPLITS:
        rel = data.get(split)
        if not isinstance(rel, str) or not rel.strip():
            raise ValueError(f"data.yaml must define local split path: {split}")
        if "://" in rel or rel.startswith(".."):
            raise ValueError(f"Split {split} must be a path under data.root, not {rel!r}")
        images = (root / rel).resolve()
        if not str(images).startswith(str(root)):
            raise ValueError(f"Split {split} resolves outside data.root: {images}")
        if not images.is_dir():
            raise FileNotFoundError(f"Missing split directory: {images}")
        labels_rel = rel.replace("images/", "labels/", 1) if "images/" in rel else f"labels/{split}"
        labels = (root / labels_rel).resolve()
        if not str(labels).startswith(str(root)):
            raise ValueError(f"Label split {split} resolves outside data.root")
        if not labels.is_dir():
            raise FileNotFoundError(f"Missing label split directory: {labels}")

    _assert_no_cross_split_images(root)
    return root


def data_yaml_for_ultralytics(root: Path, staging_dir: Path) -> Path:
    """Ultralytics reads ``path`` inside data.yaml; patch stale build-machine paths when needed."""
    root = root.resolve()
    src = root / "data.yaml"
    data = load_yaml(src)
    yaml_path = data.get("path")
    declared = (
        _resolve_data_yaml_path_field(str(yaml_path), src)
        if yaml_path is not None
        else root
    )
    if declared == root:
        return src
    staging_dir.mkdir(parents=True, exist_ok=True)
    out = staging_dir / "data.yaml"
    patched = dict(data)
    patched["path"] = root.as_posix()
    save_yaml(patched, out)
    return out


def _assert_no_cross_split_images(root: Path) -> None:
    from vn_tsd.pipelines.faster_rcnn.dataset import EXTENSIONS, file_hash

    seen: dict[str, str] = {}
    for split in SPLITS:
        image_dir = root / "images" / split
        for path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in EXTENSIONS):
            digest = file_hash(path)
            if digest in seen and seen[digest] != split:
                raise ValueError(
                    f"Byte-identical image across splits: {seen[digest]} and {split}"
                )
            seen[digest] = split
