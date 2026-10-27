"""Adapt Ultralytics DDP setup to the group owned by our launcher context."""
import os

import torch
import torch.distributed as dist
from ultralytics.models.yolo.detect.train import DetectionTrainer


class LauncherDetectionTrainer(DetectionTrainer):
    def _setup_ddp(self):
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

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode == "val":
            batch_size = int(getattr(self.args, "eval_batch_size", batch_size))
        return super().get_dataloader(dataset_path, batch_size, rank, mode)
