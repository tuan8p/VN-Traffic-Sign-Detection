from vn_tsd.runtime.train_log import METRIC_KEYS, format_epoch_line, history_row, log_early_stop

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
    row = history_row(2, 1.0, 0.5, {"mAP50-95": 0.1}, 0.01)
    assert list(row) == ["epoch", "train_loss", "val_loss", "lr", "mAP50-95", "mAP50", "mAP75", "AR100"]
    assert row["mAP50"] == 0.0

def test_early_stop_line():
    assert log_early_stop(10, 10, is_main=False) == "[train] early stopping at epoch 10 (patience=10)"
