"""Rank-zero tracking. No notebook objects are passed to DDP workers."""
from pathlib import Path

from vn_tsd.eval.metrics import summarize_yolo
from vn_tsd.utils.io import save_json


def remove_native_wandb(trainer):
    for event, handlers in trainer.callbacks.items():
        trainer.callbacks[event] = [fn for fn in handlers
            if fn.__module__ != 'ultralytics.utils.callbacks.wb']


class Tracking:
    def __init__(self, cfg, run_dir, main):
        self.cfg, self.run_dir, self.main = cfg, Path(run_dir), main
        self.run = None
        self.pending = set()
        self.history = []

    def start(self):
        if not self.main or not self.cfg['train'].get('require_wandb', False):
            return
        import os
        import wandb
        if wandb.run is not None:
            raise RuntimeError('Finish the existing W&B run first')
        if os.environ.get('WANDB_MODE', 'online') != 'online':
            raise ValueError('Required W&B tracking needs online mode')
        project = self.cfg['project']
        self.run = wandb.init(
            entity=project['wandb_entity'], project=project['wandb_project'],
            name=self.cfg['train']['run_name'], config=self.cfg,
            dir=str(self.run_dir.resolve()), mode='online',
            settings=wandb.Settings(base_url=project['wandb_base_url']),
        )
        self.run.define_metric('epoch')
        for prefix in ('train/*', 'val/*', 'lr/*'):
            self.run.define_metric(prefix, step_metric='epoch')
        save_json({'id': self.run.id, 'url': self.run.url}, self.run_dir / 'wandb_run.json')
        print('W&B training charts:', self.run.url, flush=True)

    def mark_epoch(self, trainer):
        if self.main:
            self.pending.add(trainer.epoch + 1)

    def log_epoch(self, trainer):
        epoch = trainer.epoch + 1
        if not self.main or epoch not in self.pending:
            return
        self.pending.remove(epoch)
        scores = summarize_yolo(trainer.validator.metrics)
        values = {'epoch': epoch}
        values.update({k: float(v) for k, v in trainer.label_loss_items(trainer.tloss, prefix='train').items()})
        values.update({k: float(v) for k, v in trainer.metrics.items()})
        values.update({f'val/{k}': v for k, v in scores.items() if v is not None})
        values.update({f'lr/{k}': float(v) for k, v in trainer.lr.items()})
        self.history.append({**values, 'val/AR100': scores['AR100']})
        save_json(self.history, self.run_dir / 'history.json')
        if self.run is not None:
            self.run.log(values)

    def finish(self, success):
        if self.run is not None:
            self.run.finish(exit_code=0 if success else 1)
            self.run = None
