from vn_tsd.runtime.train_log import METRIC_KEYS, format_epoch_line, history_row, log_early_stop, wandb_epoch_payload

def test_epoch_line_matches_both_pipelines():
    metrics = {"mAP50-95": 0.41, "mAP50": 0.72, "mAP75": 0.45, "AR100": 0.55}
    line = format_epoch_line(1, 20, 1.23456, 0.98765, metrics, 0.005, is_best=True)
    assert line == (
        "[train] Epoch  1/20 | train_loss: 1.2346 | val_loss: 0.9877 | "
        "mAP50-95: 0.4100 | mAP50: 0.7200 | mAP75: 0.4500 | AR100: 0.5500 | "
        "lr: 5.00e-03 -> [BEST VAL mAP50-95: 0.4100]"
    )
    assert [key for key in METRIC_KEYS if key in line] == list(METRIC_KEYS)

def test_history_row_keys():
    row = history_row(2, 1.0, 0.5, {"mAP50-95": 0.1}, 0.01, is_best=False, patience_count=3)
    assert list(row) == [
        "epoch", "train_loss", "val_loss", "lr", "is_best", "patience_count",
        "mAP50-95", "mAP50", "mAP75", "AR100",
    ]
    assert row["patience_count"] == 3
    assert row["mAP50"] is None
    # Đồng bộ test logger dùng chung theo METRICS_CONTRACT.md ở nhánh yolo:
    assert row["mAP50"] is None # input thiếu mAP50 phải giữ None (JSON null), không giả định đã đo được 0.

def test_unavailable_metric_is_na_not_zero():
    # Metric thiếu hiển thị N/A; metric thực sự được cung cấp là 0 vẫn in 0.0000.
    line = format_epoch_line(1, 2, 1., .5, {"mAP50-95": 0.}, .01)
    assert "mAP50: N/A" in line and "mAP50-95: 0.0000" in line

def test_early_stop_line():
    assert log_early_stop(10, 10, is_main=False) == "[train] early stopping at epoch 10 (patience=10)"


def test_wandb_payload_omits_missing_ar100():
    row = history_row(1, 1.0, 0.5, {"mAP50-95": 0.3}, 0.01, is_best=True, patience_count=0)
    payload = wandb_epoch_payload(row)
    assert payload["val/mAP50-95"] == 0.3
    assert "val/AR100" not in payload
