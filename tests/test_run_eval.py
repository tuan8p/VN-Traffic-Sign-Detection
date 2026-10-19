from unittest.mock import patch

import pytest

from tools.run_eval import main
from vn_tsd.utils.io import load_yaml, save_yaml


@pytest.mark.parametrize(
    ("name", "target"),
    [
        (
            "yolo26m",
            "vn_tsd.pipelines.yolo.train.YOLOPipeline",
        ),
        (
            "faster_rcnn",
            "vn_tsd.pipelines.faster_rcnn.train.FasterRCNNPipeline",
        ),
    ],
)
def test_eval_uses_saved_config(tmp_path, name, target):
    cfg = {
        "pipeline": name,
        "data": {"root": "/saved/dataset"},
        "seed": 42,
    }
    save_yaml(cfg, tmp_path / "resolved_config.yaml")

    metrics = {
        "mAP50-95": 0.4,
        "mAP50": 0.7,
        "mAP75": 0.45,
    }

    with patch(target) as pipeline_class:
        pipeline_class.return_value.evaluate.return_value = metrics

        main([
            "--pipeline", name,
            "--run-dir", str(tmp_path),
        ])

        # Truyền nguyên config đã lưu.
        pipeline_class.assert_called_once_with(
            cfg, tmp_path.resolve()
        )
        pipeline_class.return_value.evaluate.assert_called_once_with()

    # File config gốc không bị thay đổi.
    assert load_yaml(tmp_path / "resolved_config.yaml") == cfg

    import json

    result = json.loads(
        (tmp_path / "test_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["metrics"] == metrics
    assert result["data_root"] == "/saved/dataset"


def test_eval_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        main([
            "--pipeline", "yolo26m",
            "--run-dir", str(tmp_path),
        ])


def test_eval_rejects_pipeline_mismatch(tmp_path):
    save_yaml(
        {"pipeline": "faster_rcnn"},
        tmp_path / "resolved_config.yaml",
    )

    with pytest.raises(ValueError):
        main([
            "--pipeline", "yolo26m",
            "--run-dir", str(tmp_path),
        ])


def test_eval_rejects_scaffold(tmp_path):
    save_yaml(
        {"pipeline": "faster_rcnn"},
        tmp_path / "resolved_config.yaml",
    )

    with patch(
        "vn_tsd.pipelines.faster_rcnn.train.FasterRCNNPipeline"
    ) as pipeline_class:
        pipeline_class.return_value.evaluate.return_value = {
            "status": "scaffold"
        }

        with pytest.raises(NotImplementedError):
            main([
                "--pipeline", "faster_rcnn",
                "--run-dir", str(tmp_path),
            ])

    assert not (tmp_path / "test_metrics.json").exists()