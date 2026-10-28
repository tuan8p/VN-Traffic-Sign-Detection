import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from frcnn_fixtures import make_cache
from vn_tsd.config.resolve import resolve_config
from vn_tsd.pipelines.yolo.train import YOLOPipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pipeline(tmp_path):
    data_root = make_cache(tmp_path / "data", train=1, val=1, test=1)

    cfg = resolve_config(
        pipeline_yaml=ROOT / "configs/pipelines/yolo26m.yaml",
        shared_yaml=ROOT / "configs/shared.yaml",
        overrides={"data": {"root": str(data_root)}},
    )

    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)

    cfg["train"]["require_wandb"] = False
    return YOLOPipeline(cfg, run_dir)


def test_readonly_data_disables_label_cache(pipeline):
    pipeline.cfg.setdefault("runtime_cfg", {})["yolo_readonly_data"] = True
    args = pipeline._build_train_args()
    assert args["cache"] is False


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
    assert args["patience"] == 0


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
    assert args["optimizer"] == "AdamW"
    assert args["lr0"] == 0.0002
    assert args["momentum"] == 0.9
    assert "eval_batch_size" not in args


def test_missing_data_root(pipeline):
    pipeline.cfg["data"]["root"] = ""

    with pytest.raises(ValueError, match="data.root"):
        pipeline._build_train_args()


def test_missing_data_yaml(pipeline):
    data_yaml = Path(pipeline.cfg["data"]["root"]) / "data.yaml"
    data_yaml.unlink()

    with pytest.raises(FileNotFoundError):
        pipeline._build_train_args()


def test_save_best_checkpoint(pipeline):
    checkpoint = pipeline.run_dir / "checkpoints" / "checkpoint_best.pt"
    checkpoint.write_bytes(b"checkpoint")
    saved = pipeline._save_checkpoint(SimpleNamespace())
    assert saved == checkpoint
    assert saved.read_bytes() == b"checkpoint"


def test_missing_best_checkpoint(pipeline):
    with pytest.raises(FileNotFoundError):
        pipeline._save_checkpoint(SimpleNamespace())


def test_fit_saves_test_metrics(pipeline, tmp_path):
    model = Mock()
    test_metrics = {
        "mAP50-95": 0.4,
        "mAP50": 0.7,
        "mAP75": 0.45,
        "AR100": None,
    }
    tracking = Mock()
    tracking.best_epoch = 1
    tracking.best_val = test_metrics
    tracking.run = None
    tracking.finish = Mock()

    def evaluate_after_checkpoint(**kwargs):
        checkpoint = pipeline.run_dir / "checkpoints" / "checkpoint_best.pt"
        checkpoint.write_bytes(b"checkpoint")
        model.train.assert_called_once()
        payload = {"metrics": test_metrics, "predictions": [], "evaluator": {}}
        return payload if kwargs.get("detailed") else test_metrics

    with (
        patch("vn_tsd.pipelines.yolo.model.build_yolo26m", return_value=model),
        patch("vn_tsd.pipelines.yolo.tracking.Tracking", return_value=tracking),
        patch.object(pipeline, "evaluate", side_effect=evaluate_after_checkpoint) as evaluate,
    ):
        metrics = pipeline.fit()

    evaluate.assert_called_once_with(detailed=True)
    assert metrics["status"] == "completed"
    assert metrics["test"] == test_metrics

    saved = json.loads(
        (pipeline.run_dir / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["test"] == test_metrics
    assert (pipeline.run_dir / "test_metrics.json").is_file()
    assert (pipeline.run_dir / "resolved_config.yaml").is_file()

def test_skip_test_when_disabled(pipeline, tmp_path):
    pipeline.cfg['eval']['auto_eval_test'] = False
    model = Mock()
    tracking = Mock()
    tracking.best_epoch = 1
    tracking.best_val = {"mAP50-95": 0.1, "mAP50": 0.2, "mAP75": 0.1, "AR100": None}
    tracking.run = None
    tracking.finish = Mock()
    (pipeline.run_dir / "checkpoints" / "checkpoint_best.pt").write_bytes(b"checkpoint")
    with patch('vn_tsd.pipelines.yolo.model.build_yolo26m', return_value=model), \
         patch('vn_tsd.pipelines.yolo.tracking.Tracking', return_value=tracking), \
         patch.object(pipeline, 'evaluate') as evaluate:
        result = pipeline.fit()
    evaluate.assert_not_called()
    assert result['test'] is None


def test_ddp_worker_only_trains(pipeline, monkeypatch):
    from vn_tsd.pipelines.yolo import distributed
    monkeypatch.setenv('WORLD_SIZE', '2')
    monkeypatch.setenv('RANK', '1')
    pipeline.cfg['eval']['auto_eval_test'] = False
    ctx = Mock(main=False, world=2)
    ctx.phase.side_effect = lambda fn: fn()
    monkeypatch.setattr(distributed, 'Context', lambda runtime: ctx)
    model = Mock()
    with patch('vn_tsd.pipelines.yolo.model.build_yolo26m', return_value=model), \
         patch.object(pipeline, 'evaluate') as evaluate, \
         patch.object(pipeline, '_save_checkpoint') as save:
        result = pipeline.fit()
    model.train.assert_called_once()
    assert model.train.call_args.kwargs['batch'] == pipeline.cfg['train']['batch_size']
    assert model.train.call_args.kwargs['device'] == [0, 1]
    evaluate.assert_not_called()
    save.assert_not_called()
    assert not (pipeline.run_dir / 'metrics.json').exists()
    assert result['status'] == 'worker_finished'
    from vn_tsd.pipelines.yolo.trainer import LauncherDetectionTrainer
    assert model.train.call_args.kwargs["trainer"] is LauncherDetectionTrainer
    ctx.close.assert_called_once()


def test_ddp_rejects_uneven_global_batch(pipeline, monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '2')
    pipeline.cfg['train']['batch_size'] = 7
    with pytest.raises(ValueError, match='divisible'):
        pipeline._build_train_args()


def test_fit_writes_failure_json_on_error(pipeline):
    model = Mock()
    model.train.side_effect = RuntimeError("boom")
    tracking = Mock()
    tracking.run = None
    tracking.finish = Mock()

    with (
        patch("vn_tsd.pipelines.yolo.model.build_yolo26m", return_value=model),
        patch("vn_tsd.pipelines.yolo.tracking.Tracking", return_value=tracking),
        pytest.raises(RuntimeError, match="boom"),
    ):
        pipeline.fit()

    doc = json.loads((pipeline.run_dir / "failure.json").read_text(encoding="utf-8"))
    assert doc["status"] == "failed"
    assert doc["pipeline"] == "yolo26m"
    assert "boom" in doc["error"]
    metrics = json.loads((pipeline.run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "failed"
