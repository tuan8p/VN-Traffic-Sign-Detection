"""Shared training log for both detection pipelines.

Same progress bar and epoch line as the classification DL pipeline,
with detection metrics instead of accuracy and macro F1.
Both teams call this. Do not print a different epoch format.

TODO [Hai team pipeline]: gọi history_row/log_epoch bằng metric VALIDATION sau
mỗi epoch; chỉ rank chính in log và ghi history.json. Metric thiếu giữ None
(JSON null), hiển thị N/A. Không đổi thành 0 để che phần chưa triển khai.
TODO [Team pipeline Faster R-CNN + preprocessing online]: chọn best/early stop
bằng validation mAP50-95 trong train loop; is_best chỉ đánh dấu để hiển thị,
Team Faster R-CNN selects best/early stop bằng validation mAP50-95 trong train loop;
is_best chỉ đánh dấu để hiển thị,
module logging này không tự lưu checkpoint hoặc điều khiển early stopping.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

from tqdm import tqdm

log = logging.getLogger(__name__)

from vn_tsd.eval.metrics import METRIC_KEYS, PRIMARY_METRIC, summarize


def train_bar(
    loader: Iterable[Any],
    epoch: int,
    total_epochs: int,
    *,
    stage: str = "train",
    is_main: bool = True,
):
    """Batch progress bar. Only the main rank shows it."""
    return tqdm(
        loader,
        desc=f"[{stage}] Epoch {epoch:2d}/{total_epochs:2d}",
        disable=not is_main,
        leave=False,
        dynamic_ncols=True,
    )


def note_loss(bar, loss: float) -> None:
    bar.set_postfix({"loss": f"{float(loss):.4f}"})


def history_row(
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: dict[str, Any],
    lr: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": epoch,
        "train_loss": round(float(train_loss), 5),
        "val_loss": round(float(val_loss), 5),
        "lr": float(lr),
    }
    for key, value in summarize(metrics).items():
        row[key] = round(value, 5) if value is not None else None
    return row


def format_epoch_line(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    metrics: dict[str, Any],
    lr: float,
    *,
    stage: str = "train",
    is_best: bool = False,
    primary: str = PRIMARY_METRIC,
) -> str:
    parts = [
        f"[{stage}] Epoch {epoch:2d}/{total_epochs:2d}",
        f"train_loss: {float(train_loss):.4f}",
        f"val_loss: {float(val_loss):.4f}",
    ]
    scores = summarize(metrics)
    for key in METRIC_KEYS:
        value = scores[key]
        display = f"{value:.4f}" if value is not None else "N/A"
        parts.append(f"{key}: {display}")
    line = " | ".join(parts) + f" | lr: {float(lr):.2e}"
    if is_best:
        value = metrics.get(primary)
        if value is None:
            raise ValueError(f"Cannot mark best checkpoint without {primary}")
        score = float(value)
        line += f" -> [BEST VAL {primary}: {score:.4f}]"
    return line


def log_epoch(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    metrics: dict[str, Any],
    lr: float,
    *,
    stage: str = "train",
    is_best: bool = False,
    primary: str = PRIMARY_METRIC,
    is_main: bool = True,
) -> str:
    line = format_epoch_line(
        epoch,
        total_epochs,
        train_loss,
        val_loss,
        metrics,
        lr,
        stage=stage,
        is_best=is_best,
        primary=primary,
    )
    if is_main:
        print(line, flush=True)
        log.info(line)
    return line


def log_early_stop(epoch: int, patience: int, *, stage: str = "train", is_main: bool = True) -> str:
    line = f"[{stage}] early stopping at epoch {epoch} (patience={patience})"
    if is_main:
        print(line, flush=True)
        log.info(line)
    return line
