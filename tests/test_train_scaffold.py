"""Shared train entry imports; full fit() tests live in test_frcnn_pipeline / test_yolo_train."""

from vn_tsd.runtime.train_log import history_row


def test_history_row_is_shared_train_log_contract():
    row = history_row(1, 1.0, 0.5, {"mAP50-95": 0.2}, 0.01, is_best=True, patience_count=0)
    assert row["is_best"] is True
    assert row["patience_count"] == 0
    assert row["mAP50-95"] == 0.2
