from pathlib import Path

import pytest
import yaml

from frcnn_fixtures import make_cache
from vn_tsd.data.dataset_contract import validate_processed_dataset


def test_valid_processed_cache(tmp_path):
    root = make_cache(tmp_path / "cache")
    cfg = {"data": {"root": str(root)}}
    assert validate_processed_dataset(cfg) == root.resolve()


def test_rejects_external_split_path(tmp_path):
    root = make_cache(tmp_path / "cache")
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    data["val"] = "https://example.com/val"
    (root / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="under data.root"):
        validate_processed_dataset({"data": {"root": str(root)}})


def test_accepts_stale_build_path_when_splits_under_root(tmp_path):
    root = make_cache(tmp_path / "cache")
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    data["path"] = "E:/BACHKHOA/28_COMPUTER_VISION/BTL/BTL3/VN-Traffic-Sign-Detection/data/processed"
    (root / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    assert validate_processed_dataset({"data": {"root": str(root)}}) == root.resolve()


def test_accepts_mismatching_absolute_path_when_splits_under_root(tmp_path):
    root = make_cache(tmp_path / "cache")
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    data["path"] = str(tmp_path / "other")
    (root / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    assert validate_processed_dataset({"data": {"root": str(root)}}) == root.resolve()
