
from unittest.mock import patch

import pytest

from vn_tsd.pipelines.yolo.model import build_yolo26m


def test_load_model_from_config():
    cfg = {
        "model": {
            "weights": "yolo26m.pt"
        }
    }

    with patch(
        "vn_tsd.pipelines.yolo.model.YOLO"
    ) as mock_yolo:
        mock_yolo.return_value.task = "detect"

        model = build_yolo26m(cfg)

        mock_yolo.assert_called_once_with(
            "yolo26m.pt",
            task="detect"
        )

        assert model.task == "detect"


def test_invalid_weights():
    cfg = {
        "model": {
            "weights": ""
        }
    }

    with pytest.raises(ValueError):
        build_yolo26m(cfg)
