from pathlib import Path
from unittest.mock import patch

import pytest

from vn_tsd.pipelines.yolo.weights import resolve_yolo_weights


def test_prefers_runtime_yolo_pretrained(tmp_path):
    pt = tmp_path / "custom.pt"
    pt.write_bytes(b"x")
    cfg = {
        "runtime_cfg": {"yolo_pretrained": str(pt)},
        "model": {"weights": "yolo26m.pt"},
    }
    assert resolve_yolo_weights(cfg) == str(pt.resolve())


def test_falls_back_to_kaggle_model_path(monkeypatch):
    kaggle_pt = Path("/kaggle/input/models/ultralytics/yolo26/pytorch/default/1/yolo26m.pt")

    class _Stub:
        def is_file(self):
            return True

        def resolve(self):
            return kaggle_pt

    monkeypatch.setattr(
        "vn_tsd.pipelines.yolo.weights.KAGGLE_YOLO26M_PT",
        _Stub(),
    )
    cfg = {"model": {"weights": "yolo26m.pt"}, "runtime_cfg": {}}
    assert resolve_yolo_weights(cfg) == str(kaggle_pt)


def test_falls_back_to_weight_name_when_no_local_file():
    cfg = {"model": {"weights": "yolo26m.pt"}, "runtime_cfg": {}}
    with patch.object(Path, "is_file", return_value=False):
        assert resolve_yolo_weights(cfg) == "yolo26m.pt"


def test_rejects_empty_weights():
    cfg = {"model": {"weights": ""}, "runtime_cfg": {}}
    with patch.object(Path, "is_file", return_value=False):
        with pytest.raises(ValueError, match="model.weights"):
            resolve_yolo_weights(cfg)
