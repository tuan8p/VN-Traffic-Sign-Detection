"""Shared training log for both detection pipelines.

Same progress bar and epoch line as the classification DL pipeline,
with detection metrics instead of accuracy and macro F1.
Both teams call this. Do not print a different epoch format.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

from tqdm import tqdm

log = logging.getLogger(__name__)

METRIC_KEYS = ("mAP50-95", "mAP50", "mAP75", "AR100")
PRIMARY_METRIC = "mAP50-95"


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
    for key in METRIC_KEYS:
        row[key] = round(float(metrics.get(key, 0.0)), 5)
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
    for key in METRIC_KEYS:
        parts.append(f"{key}: {float(metrics.get(key, 0.0)):.4f}")
    line = " | ".join(parts) + f" | lr: {float(lr):.2e}"
    if is_best:
        score = float(metrics.get(primary, 0.0))
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
