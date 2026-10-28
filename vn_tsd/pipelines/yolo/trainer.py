"""Adapt Ultralytics DDP setup to the group owned by our launcher context."""
import os

import torch
import torch.distributed as dist

try:
    from ultralytics.models.yolo.detect.train import DetectionTrainer
except ImportError:  # pragma: no cover - allows unit testing without ultralytics installed
    class DetectionTrainer:  # type: ignore[no-redef]
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            raise RuntimeError("ultralytics is required for DetectionTrainer")


class LauncherDetectionTrainer(DetectionTrainer):
    def _setup_ddp(self, world_size=None):
        if not dist.is_initialized():
            raise RuntimeError('Launcher DDP process group must be initialized before training')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        expected_world = world_size if world_size is not None else getattr(self, 'world_size', None)
        if dist.get_rank() != rank or (expected_world is not None and dist.get_world_size() != expected_world):
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

