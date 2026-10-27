"""Rank-zero training log aligned with vn_tsd.runtime.train_log."""
from __future__ import annotations

import math
import shutil
from pathlib import Path

from vn_tsd.eval.metrics import PRIMARY_METRIC
from vn_tsd.pipelines.yolo.coco_eval import evaluate_split_coco
from vn_tsd.runtime.train_log import history_row, log_early_stop, log_epoch, wandb_epoch_payload
from vn_tsd.utils.io import save_json


def remove_native_wandb(trainer):
    for event, handlers in trainer.callbacks.items():
        trainer.callbacks[event] = [fn for fn in handlers
            if fn.__module__ != 'ultralytics.utils.callbacks.wb']


class Tracking:
    def __init__(self, cfg, run_dir, main):
        self.cfg, self.run_dir, self.main = cfg, Path(run_dir), main
        self.run = None
        self.pending: set[int] = set()
        self.history: list[dict] = []
        self._best_score = -math.inf
        self._bad_epochs = 0
        self.best_epoch: int | None = None
        self.best_val: dict | None = None
        self._patience = int(cfg.get("train", {}).get("early_stopping_patience", 0))
        self._total_epochs = int(cfg.get("train", {}).get("epochs", 1))
        self._stopped_early = False

    def start(self):
        if not self.main or not self.cfg['train'].get('require_wandb', False):
            return
        import os
        import wandb
        if wandb.run is not None:
            raise RuntimeError('Finish the existing W&B run first')
        if os.environ.get('WANDB_MODE', 'online') != 'online':
            raise ValueError('Required W&B tracking needs online mode')
        from vn_tsd.runtime.wandb_urls import wandb_api_base_url, wandb_run_ui_url

        project = self.cfg['project']
        api_base = wandb_api_base_url(project)
        self.run = wandb.init(
            entity=project['wandb_entity'], project=project['wandb_project'],
            name=self.cfg['train']['run_name'], config=self.cfg,
            dir=str(self.run_dir.resolve()), mode='online',
            settings=wandb.Settings(base_url=api_base),
        )
        self.run.define_metric('epoch')
        for prefix in ('train_loss', 'val_loss', 'lr', 'val/*', 'is_best', 'patience_count'):
            self.run.define_metric(prefix, step_metric='epoch')
        ui_url = wandb_run_ui_url(project, self.run.id)
        save_json({'id': self.run.id, 'url': ui_url, 'api_base_url': api_base}, self.run_dir / 'wandb_run.json')
        print('W&B training charts:', ui_url, flush=True)

    def mark_epoch(self, trainer):
        if self.main:
            self.pending.add(trainer.epoch + 1)

    def log_epoch(self, trainer):
        epoch = trainer.epoch + 1
        if not self.main or epoch not in self.pending:
            return
        self.pending.remove(epoch)
        val_metrics = evaluate_split_coco(self.cfg, Path(trainer.last), "val")["metrics"]
        train_loss = _train_loss(trainer)
        val_loss = _val_loss(trainer)
        lr = _learning_rate(trainer)
        score = val_metrics.get(PRIMARY_METRIC)
        if score is None or not math.isfinite(score):
            raise ValueError("Validation mAP50-95 is undefined; cannot select best checkpoint")
        improved = score > self._best_score
        if improved:
            self._best_score = float(score)
            self._bad_epochs = 0
            self.best_epoch = epoch
            self.best_val = dict(val_metrics)
            _save_best_from_trainer(trainer, self.run_dir)
        else:
            self._bad_epochs += 1
        row = history_row(
            epoch, train_loss, val_loss, val_metrics, lr,
            is_best=improved, patience_count=self._bad_epochs,
        )
        self.history.append(row)
        save_json(self.history, self.run_dir / 'history.json')
        log_epoch(
            epoch, self._total_epochs, train_loss, val_loss, val_metrics, lr,
            is_best=improved,
        )
        if self.run is not None:
            payload = wandb_epoch_payload(row)
            payload["epoch"] = epoch
            self.run.log(payload, step=epoch)
        if self._patience > 0 and self._bad_epochs >= self._patience:
            trainer.stop = True
            self._stopped_early = True
            log_early_stop(epoch, self._patience)

    def on_train_end(self, trainer):
        if not self.main or not self._stopped_early:
            return
        if trainer.epoch + 1 >= self._total_epochs:
            self._stopped_early = False

    def finish(self, success):
        if self.run is not None:
            self.run.finish(exit_code=0 if success else 1)
            self.run = None


def _save_best_from_trainer(trainer, run_dir: Path) -> None:
    last = Path(trainer.last)
    if not last.is_file():
        raise FileNotFoundError(f"Expected trainer checkpoint after epoch save: {last}")
    dest = run_dir / "checkpoints" / "checkpoint_best.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if last.resolve() != dest.resolve():
        shutil.copy2(last, dest)


def _train_loss(trainer) -> float:
    items = trainer.label_loss_items(trainer.tloss, prefix='train')
    return float(sum(items.values()))


def _val_loss(trainer) -> float:
    metrics = trainer.metrics
    if 'val/box_loss' in metrics:
        return float(metrics['val/box_loss'])
    val_items = {k: float(v) for k, v in metrics.items() if str(k).startswith('val/') and 'loss' in str(k)}
    if val_items:
        return float(sum(val_items.values()))
    raise ValueError('Ultralytics trainer did not expose validation box loss')


def _learning_rate(trainer) -> float:
    if not trainer.lr:
        raise ValueError('Ultralytics trainer did not expose learning rate')
    return float(next(iter(trainer.lr.values())))
