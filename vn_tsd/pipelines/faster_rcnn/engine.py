"""Training arithmetic separated from orchestration for focused tests."""
from __future__ import annotations

import math
import random

import numpy as np
import torch

from vn_tsd.runtime.train_log import train_bar, note_loss
from .evaluate import LOSS_KEYS, to_device


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class WarmupCosine:
    """LR for the next successful update, not for an AMP-skipped attempt."""
    def __init__(self, optimizer, cfg, steps):
        if steps < 1:
            raise ValueError("Training loader must have at least one step")
        self.optimizer, self.cfg, self.steps, self.updates = optimizer, cfg, steps, 0
        self.apply()

    def value(self, u):
        t = self.cfg["epochs"]*self.steps
        # Short smoke runs may end inside the configured warmup period.
        w = min(self.cfg.get("warmup_epochs", 1), self.cfg["epochs"])*self.steps
        base, f, minimum = self.cfg["lr"], self.cfg["warmup_start_factor"], self.cfg["min_lr"]
        if w == t:
            return base*(f+(1-f)*min(u/w, 1))
        if u < w:
            return base*(f+(1-f)*u/w)
        if u >= t:
            return minimum
        return minimum+(base-minimum)*(1+math.cos(math.pi*(u-w)/(t-w)))/2

    def apply(self):
        for group in self.optimizer.param_groups:
            group["lr"] = self.value(self.updates)

    def step(self):
        self.updates += 1
        self.apply()


class BestTracker:
    def __init__(self, patience):
        self.patience, self.best, self.bad, self.epoch = patience, -math.inf, 0, None

    def update(self, score, epoch):
        if score is None or not math.isfinite(score):
            raise ValueError("Validation AP is undefined; cannot select best checkpoint")
        improved = score > self.best
        if improved:
            self.best, self.bad, self.epoch = score, 0, epoch
        else:
            self.bad += 1
        return improved, self.bad >= self.patience


def train_epoch(model, loader, optimizer, scaler, schedule, ctx, epoch, epochs, amp):
    model.train()
    sums, count, successful, skipped, last_lr = [0.0]*4, 0, 0, 0, None
    bar = train_bar(loader, epoch, epochs, is_main=ctx.main)
    # Batch read errors occur before DDP forward, so report them at a common boundary.
    iterator = iter(bar)
    for _ in range(len(loader)):
        images, targets = ctx.phase(lambda: next(iterator))
        images, targets = to_device(images, targets, ctx.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=ctx.device.type, dtype=torch.float16, enabled=amp):
            losses = model(images, targets)
            if set(losses) != set(LOSS_KEYS):
                raise ValueError("Expected four detection loss components")
            loss = sum(losses.values())
        finite = bool(torch.isfinite(loss).item())
        if not all(ctx.gather(finite)):
            raise RuntimeError("Nonfinite training loss on a rank; abort before backward")
        scaler.scale(loss).backward()
        used_lr = optimizer.param_groups[0]["lr"]
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        updated = not amp or scaler.get_scale() >= old_scale
        if len(set(ctx.gather(updated))) != 1:
            raise RuntimeError("AMP optimizer update mismatch between ranks")
        if updated:
            last_lr = used_lr
            schedule.step()
            successful += 1
        else:
            skipped += 1
        for i, k in enumerate(LOSS_KEYS):
            sums[i] += float(losses[k].detach())*len(images)
        count += len(images)
        if ctx.main:
            note_loss(bar, float(loss.detach()))
    bar.close()
    if successful == 0:
        raise RuntimeError(f"No successful optimizer updates in epoch {epoch}; AMP skipped={skipped}")
    values = ctx.sum(sums+[count])
    components = {k: values[i]/values[-1] for i, k in enumerate(LOSS_KEYS)}
    return dict(loss=sum(components.values()), components=components, lr=last_lr,
                successful_updates=successful, amp_skipped_updates=skipped)
