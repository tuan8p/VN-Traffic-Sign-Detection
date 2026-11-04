"""Mild image-level rare-class oversampling, deterministic across DDP ranks."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler


def image_sampling_weights(labels, num_classes, power=0.25, max_weight=2.0):
    present = []
    counts = np.zeros(num_classes, dtype=np.int64)
    for label in labels:
        classes = np.asarray(label['cls']).reshape(-1)
        if not np.isfinite(classes).all() or np.any(classes != classes.astype(int)):
            raise ValueError('Sampling requires finite integer class IDs')
        ids = np.unique(classes.astype(int))
        if np.any(ids < 0) or np.any(ids >= num_classes):
            raise ValueError('Sampling class ID outside dataset class map')
        present.append(ids)
        counts[ids] += 1  # image frequency, not number of objects
    if not present:
        raise ValueError('Cannot sample an empty dataset')
    maximum = max(int(counts.max()), 1)
    class_weights = (maximum / np.maximum(counts, 1)) ** power
    weights = [min(float(class_weights[ids].max()), max_weight) if len(ids) else 1.0
               for ids in present]
    return torch.tensor(weights, dtype=torch.double)


class ClassAwareSampler(DistributedSampler):
    """Draw one global replacement sample stream then shard it by rank.

    Epoch length stays ceil(N/world_size) per rank. Repeated images are intended;
    validation/test keep their original sampling. No extra images are created.
    """
    def __init__(self, dataset, weights, *, rank=0, world_size=1, seed=42):
        super().__init__(dataset, num_replicas=world_size, rank=rank, seed=seed)
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if len(self.weights) != len(dataset) or not torch.isfinite(self.weights).all() or (self.weights <= 0).any():
            raise ValueError('Sampling weights must be positive, finite and match dataset length')

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=generator).tolist()
        return iter(indices[self.rank:self.total_size:self.num_replicas])


class EpochDataLoader(DataLoader):
    # A new iterator/workers each epoch avoids InfiniteDataLoader prefetching the
    # next weighted stream before set_epoch(). These hooks are used by Ultralytics.
    def reset(self):
        pass

    def close(self):
        pass


def class_aware_loader(loader, policy, *, num_classes, rank, world_size, seed):
    weights = image_sampling_weights(loader.dataset.labels, num_classes,
                                     power=policy['power'], max_weight=policy['max_weight'])
    sampler = ClassAwareSampler(loader.dataset, weights, rank=rank, world_size=world_size, seed=seed)
    loader.close()
    return EpochDataLoader(loader.dataset, batch_size=loader.batch_size, sampler=sampler,
                           num_workers=loader.num_workers, pin_memory=loader.pin_memory,
                           collate_fn=loader.collate_fn, worker_init_fn=loader.worker_init_fn,
                           generator=loader.generator, drop_last=loader.drop_last)


def set_sampling_epoch(trainer):
    # Ultralytics only calls set_epoch automatically in DDP. Also advance the
    # weighted stream for single-GPU runs so every epoch gets new draws.
    sampler = trainer.train_loader.sampler
    if isinstance(sampler, ClassAwareSampler):
        sampler.set_epoch(trainer.epoch)
