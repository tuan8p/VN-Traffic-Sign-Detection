import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vn_tsd.config.resolve import resolve_config
from vn_tsd.pipelines.yolo.train import YOLOPipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pipeline(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    # Chỉ kiểm tra arguments, không dùng dataset này để train.
    (data_root / "data.yaml").write_text("# test\n", encoding="utf-8")

    cfg = resolve_config(
        pipeline_yaml=ROOT / "configs/pipelines/yolo26m.yaml",
        shared_yaml=ROOT / "configs/shared.yaml",
        overrides={"data": {"root": str(data_root)}},
    )

    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)

    return YOLOPipeline(cfg, run_dir)


def test_online_off_disables_augmentation(pipeline):
    pipeline.cfg["online_preprocess"]["enabled"] = False

    args = pipeline._build_train_args()

    for key in (
        "mosaic", "mixup", "hsv_h", "hsv_s", "hsv_v",
        "translate", "scale", "degrees", "shear", "perspective",
        "fliplr", "flipud",
    ):
        assert args[key] == 0.0

    assert args["batch"] == pipeline.cfg["train"]["batch_size"]


def test_online_on_applies_flags(pipeline):
    pipeline.cfg["online_preprocess"].update({
        "enabled": True,
        "mosaic": True,
        "mixup": False,
        "color_jitter": False,
        "jitter": 0.2,
    })

    args = pipeline._build_train_args()

    assert args["mosaic"] == 1.0
    assert args["mixup"] == 0.0
    assert args["hsv_s"] == 0.0
    assert args["translate"] == 0.2
    assert args["scale"] == 0.2


def test_missing_data_root(pipeline):
    pipeline.cfg["data"]["root"] = ""

    with pytest.raises(ValueError, match="data.root"):
        pipeline._build_train_args()


def test_missing_data_yaml(pipeline):
    data_yaml = Path(pipeline.cfg["data"]["root"]) / "data.yaml"
    data_yaml.unlink()

    with pytest.raises(FileNotFoundError):
        pipeline._build_train_args()


def test_save_best_checkpoint(pipeline, tmp_path):
    source = tmp_path / "best.pt"
    source.write_bytes(b"checkpoint")

    model = SimpleNamespace(
        trainer=SimpleNamespace(best=source)
    )

    saved = pipeline._save_checkpoint(model)

    assert saved == (
        pipeline.run_dir / "checkpoints" / "checkpoint_best.pt"
    )
    assert saved.read_bytes() == b"checkpoint"


def test_missing_best_checkpoint(pipeline, tmp_path):
    model = SimpleNamespace(
        trainer=SimpleNamespace(best=tmp_path / "missing.pt")
    )

    with pytest.raises(FileNotFoundError):
        pipeline._save_checkpoint(model)


def test_fit_saves_test_metrics(pipeline, tmp_path):
    source = tmp_path / "best.pt"
    source.write_bytes(b"checkpoint")

    model = Mock()
    model.trainer.best = source

    test_metrics = {
        "mAP50-95": 0.4,
        "mAP50": 0.7,
        "mAP75": 0.45,
    }

    def evaluate_after_checkpoint():
        checkpoint = (
            pipeline.run_dir / "checkpoints" / "checkpoint_best.pt"
        )
        assert checkpoint.read_bytes() == b"checkpoint"
        model.train.assert_called_once()
        return test_metrics

    with (
        patch(
            "vn_tsd.pipelines.yolo.train.build_yolo26m",
            return_value=model,
        ),
        patch.object(
            pipeline,
            "evaluate",
            side_effect=evaluate_after_checkpoint,
        ) as evaluate,
    ):
        metrics = pipeline.fit()

    evaluate.assert_called_once_with()
    assert metrics["status"] == "trained"
    assert metrics["test_metrics"] == test_metrics

    saved = json.loads(
        (pipeline.run_dir / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["test_metrics"] == test_metrics
    assert (pipeline.run_dir / "resolved_config.yaml").is_file()