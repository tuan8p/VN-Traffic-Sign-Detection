from types import SimpleNamespace

import pytest

from vn_tsd.eval.metrics import summarize, summarize_yolo
from vn_tsd.runtime.train_log import history_row, format_epoch_line


def test_yolo_does_not_substitute_mean_recall_for_ar100():
    result = SimpleNamespace(box=SimpleNamespace(
        map=0.4, map50=0.7, map75=0.45, mr=0.99,
    ))
    metrics = summarize_yolo(result)
    assert metrics == {
        "mAP50-95": 0.4, "mAP50": 0.7, "mAP75": 0.45, "AR100": None,
    }
    assert history_row(1, 1, 1, metrics, 0.01)["AR100"] is None
    assert "AR100: N/A" in format_epoch_line(1, 2, 1, 1, metrics, 0.01)


@pytest.mark.parametrize("value", [-1, 1.1, float('nan'), float('inf'), True])
def test_invalid_metric_rejected(value):
    with pytest.raises(ValueError):
        summarize({"mAP50-95": value})


def test_missing_differs_from_zero():
    assert summarize({"mAP50": 0})["mAP50"] == 0
    assert summarize({})["mAP50"] is None


def test_best_requires_primary_metric():
    with pytest.raises(ValueError):
        format_epoch_line(1, 2, 1, 1, {}, 0.01, is_best=True)
