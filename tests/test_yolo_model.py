import sys
from types import ModuleType
from unittest.mock import Mock, patch

from vn_tsd.pipelines.yolo.model import build_yolo26m


def _fake_ultralytics():
    mod = ModuleType("ultralytics")
    mod.YOLO = Mock()
    return mod


def test_load_model_from_config():
    cfg = {
        "model": {
            "weights": "yolo26m.pt"
        }
    }

    fake = _fake_ultralytics()
    fake.YOLO.return_value.task = "detect"
    with (
        patch.dict(sys.modules, {"ultralytics": fake}),
        patch(
            "vn_tsd.pipelines.yolo.weights.resolve_yolo_weights",
            return_value="yolo26m.pt",
        ),
    ):
        model = build_yolo26m(cfg)

        fake.YOLO.assert_called_once_with("yolo26m.pt", task="detect")
        assert model.task == "detect"


