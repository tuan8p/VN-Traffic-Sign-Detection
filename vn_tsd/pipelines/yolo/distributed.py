"""YOLO launcher context; Ultralytics owns model wrapping and training collectives."""
from datetime import timedelta
import os

import torch
import torch.distributed as dist


class Context:
    def __init__(self, runtime=None):
        # Rank 1 can wait while rank 0 scans the full dataset before a barrier.
        timeout_seconds = (runtime or {}).get('ddp_timeout_seconds', 1800)
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
            raise ValueError('runtime_cfg.ddp_timeout_seconds must be a positive integer')
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
                dist.init_process_group('nccl', timeout=timedelta(seconds=timeout_seconds))
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
        if self.world <= 1 or not dist.is_initialized():
            return
        try:
            dist.barrier()
        except Exception:
            pass
        if self.owns_group:
            try:
                dist.destroy_process_group()
            except Exception:
                pass
