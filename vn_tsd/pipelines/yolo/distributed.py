"""YOLO launcher context; Ultralytics owns model wrapping and training collectives."""
from datetime import timedelta
import os

import torch
import torch.distributed as dist


class Context:
    def __init__(self):
        self.rank = int(os.environ.get('RANK', '0'))
        self.world = int(os.environ.get('WORLD_SIZE', '1'))
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self.main = self.rank == 0
        self.owns_group = False
        if self.world > 1:
            if not 0 <= self.rank < self.world:
                raise ValueError('Invalid distributed rank')
            if not torch.cuda.is_available() or torch.cuda.device_count() < self.world:
                raise RuntimeError('DDP requires one visible CUDA GPU per worker')
            torch.cuda.set_device(self.local_rank)
            if not dist.is_initialized():
                dist.init_process_group('nccl', timeout=timedelta(minutes=5))
                self.owns_group = True

    def phase(self, fn):
        """Share controlled failures before entering the next training collective."""
        result, error = None, None
        try:
            result = fn()
        except Exception as exc:
            error = f'rank {self.rank}: {type(exc).__name__}: {exc}'
        if self.world > 1:
            errors = [None] * self.world
            dist.all_gather_object(errors, error)
        else:
            errors = [error]
        if any(errors):
            raise RuntimeError('; '.join(e for e in errors if e))
        return result

    def close(self):
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()
