"""COCO export handles clipped boxes without concealing corrupt predictions."""
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from ultralytics.engine.results import Boxes
from ultralytics.utils.ops import clip_boxes

from frcnn_fixtures import make_cache
from vn_tsd.config.resolve import resolve_config
from vn_tsd.pipelines.faster_rcnn.dataset import load_cache
from vn_tsd.pipelines.yolo.coco_eval import _predictions_coco
from vn_tsd.pipelines.yolo.tracking import Tracking


THRESHOLDS = dict(score_thresh=0.001, nms_thresh=0.5, detections_per_img=100)


def prediction_model(rows):
    boxes = Boxes(torch.tensor(rows, dtype=torch.float32).reshape(-1, 6), orig_shape=(640, 640))
    model = Mock()
    model.predict.return_value = [SimpleNamespace(boxes=boxes)]
    return model


def test_export_skips_boxes_collapsed_by_clipping_and_keeps_small_boxes(caplog):
    data = torch.tensor([
        [-50., 10., -10., 20., .8, 0.],
        [10., 650., 20., 700., .8, 0.],
        [10., 10., 10.5, 10.5, .9, 51.],
    ])
    clip_boxes(data[:, :4], (640, 640))
    model = prediction_model(data.tolist())
    dataset = SimpleNamespace(records=[dict(path=Path('val/sign.png'), image_id=42)])
    with caplog.at_level(logging.WARNING):
        predictions = _predictions_coco(model, dataset, THRESHOLDS)
    assert len(predictions) == 1
    assert predictions[0]['bbox'] == [10., 10., .5, .5]
    assert predictions[0]['image_id'] == 42
    assert predictions[0]['category_id'] == 51
    assert len(caplog.records) == 1
    assert 'skipped 2 zero-area' in caplog.text
    assert 'val/sign.png' in caplog.text


@pytest.mark.parametrize('row,reason', [
    ([float('nan'), 0, 10, 10, .9, 0], 'Non-finite'),
    ([0, 0, float('inf'), 10, .9, 0], 'Non-finite'),
    ([0, 0, 10, 10, float('nan'), 0], 'Non-finite'),
    ([0, 0, 10, 10, .9, float('nan')], 'Non-finite'),
    ([0, 0, 10, 10, .9, float('inf')], 'Non-finite'),
    ([0, 0, 10, 10, -.1, 0], 'confidence'),
    ([0, 0, 10, 10, 1.1, 0], 'confidence'),
    ([0, 0, 10, 10, .9, -1], 'class ID'),
    ([0, 0, 10, 10, .9, 52], 'class ID'),
    ([0, 0, 10, 10, .9, .5], 'class ID'),
    ([20, 0, 10, 10, .9, 0], 'Reversed'),
    ([0, 20, 10, 10, .9, 0], 'Reversed'),
    # Even a zero-area box must not conceal a bad class ID.
    ([0, 0, 0, 10, .9, 52], 'class ID'),
])
def test_invalid_predictions_raise_with_image_and_values(row, reason):
    dataset = SimpleNamespace(records=[dict(path=Path('val/sign.png'), image_id=42)])
    with pytest.raises(ValueError, match=reason) as exc:
        _predictions_coco(prediction_model([row]), dataset, THRESHOLDS)
    for detail in ('image_id=42', 'val/sign.png', 'bbox=', 'score=', 'class_id='):
        assert detail in str(exc.value)


def test_zero_area_boxes_can_leave_an_empty_prediction_set(caplog):
    dataset = SimpleNamespace(records=[dict(path=Path('val/sign.png'), image_id=42)])
    predictions = _predictions_coco(prediction_model([[0, 0, 0, 10, .9, 0]]), dataset, THRESHOLDS)
    assert predictions == []
    assert 'skipped 1 zero-area' in caplog.text


def test_filtered_boxes_do_not_shift_image_ids_between_batches():
    dataset = SimpleNamespace(records=[
        dict(path=Path('val/first.png'), image_id=7),
        dict(path=Path('val/second.png'), image_id=42),
    ])
    model = prediction_model([[0, 0, 0, 10, .9, 0]])
    first_result = model.predict.return_value
    second_result = prediction_model([[10, 10, 20, 20, .9, 0]]).predict.return_value
    model.predict.side_effect = [first_result, second_result]
    predictions = _predictions_coco(model, dataset, THRESHOLDS, batch_size=1)
    assert len(predictions) == 1
    assert predictions[0]['image_id'] == 42


def test_epoch_11_finishes_coco_tracking_with_zero_area_prediction(tmp_path):
    root = make_cache(tmp_path / 'data', train=1, val=1, test=1)
    repo = Path(__file__).resolve().parents[1]
    cfg = resolve_config(repo / 'configs/pipelines/yolo26m.yaml', repo / 'configs/shared.yaml',
                         overrides={'data': {'root': str(root)}})
    datasets, _ = load_cache(cfg, splits=('val',))
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    last = run_dir / 'last.pt'
    last.write_bytes(b'checkpoint')
    trainer = SimpleNamespace(
        epoch=10, last=last, tloss=None, stop=False,
        label_loss_items=lambda *args, **kwargs: {'train/box_loss': 1.},
        metrics={'val/box_loss': 1.}, lr={'pg0': .0002},
    )
    tracking = Tracking(cfg, run_dir, main=True)
    tracking._val_dataset = datasets['val']
    model = prediction_model([[640, 10, 640, 20, .9, 0], [240, 240, 400, 400, .9, 0]])
    with patch('ultralytics.YOLO', return_value=model):
        tracking.mark_epoch(trainer)
        tracking.log_epoch(trainer)
    assert tracking.best_epoch == 11
    assert tracking.history[-1]['epoch'] == 11
    assert tracking.history[-1]['mAP50-95'] == pytest.approx(1.)
    assert (run_dir / 'checkpoints/checkpoint_best.pt').read_bytes() == b'checkpoint'
