"""Adapt Ultralytics DDP setup to the group owned by our launcher context."""
import os

import torch
import torch.distributed as dist
from ultralytics.models.yolo.detect.train import DetectionTrainer


class LauncherDetectionTrainer(DetectionTrainer):
    def _setup_ddp(self):
        # Reinitializing the default group after destroy can reuse stale rendezvous
        # state. Keep the launcher's group alive through setup, train and finalization.
        if not dist.is_initialized():
            raise RuntimeError('Launcher DDP process group must be initialized before training')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        if dist.get_rank() != rank or dist.get_world_size() != self.world_size:
            raise RuntimeError('Trainer and launcher rank/world size do not match')
        if dist.get_backend() != 'nccl':
            raise RuntimeError('YOLO CUDA DDP requires NCCL')
        self.device = torch.device('cuda', local_rank)
        self.accelerator = torch.cuda
        self.accelerator.set_device(local_rank)
        os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
