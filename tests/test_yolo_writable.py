from pathlib import Path

from vn_tsd.data import yolo_writable
from vn_tsd.data.yolo_writable import prepare_yolo_writable_dataset
from vn_tsd.utils.io import load_yaml, save_yaml


def _mini_cache(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / "a.jpg").write_bytes(b"img")
        (root / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    save_yaml({"path": "/old", "nc": 1, "names": ["x"]}, root / "data.yaml")
    (root / "meta.json").write_text('{"synthetic": true}', encoding="utf-8")


def test_full_copy_when_not_kaggle_input(tmp_path):
    source = tmp_path / "input"
    working = tmp_path / "working"
    _mini_cache(source)
    out = prepare_yolo_writable_dataset(source, working)
    assert out.parent == working
    assert (out / "images" / "train" / "a.jpg").read_bytes() == b"img"
    (out / "labels" / "train.cache").write_text("cache")
    assert not (source / "labels" / "train.cache").exists()


def test_stage_symlinks_images_and_copies_labels(tmp_path):
    source = tmp_path / "processed"
    working = tmp_path / "working"
    fixed = working / "yolo_dataset_test"
    _mini_cache(source)
    yolo_writable._stage_kaggle_input(source, fixed)
    yolo_writable._patch_data_yaml(fixed)
    assert (fixed / "images" / "train").is_symlink() or (fixed / "images" / "train" / "a.jpg").is_file()
    assert (fixed / "labels" / "train" / "a.txt").is_file()
    (fixed / "labels" / "train.cache").write_text("ok")
    assert load_yaml(fixed / "data.yaml")["path"] == str(fixed.resolve())


def test_reuses_existing_staged_tree(tmp_path):
    source = tmp_path / "input"
    working = tmp_path / "working"
    _mini_cache(source)
    first = prepare_yolo_writable_dataset(source, working)
    second = prepare_yolo_writable_dataset(source, working)
    assert first == second
