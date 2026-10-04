"""Global batch in config, per-rank batch for DDP.

configs/shared.yaml train.batch_size and eval.batch_size are the total
across all GPUs. A torchvision DataLoader on each rank must use
per_rank_batch. Ultralytics YOLO takes the total and splits it itself.
"""
from __future__ import annotations
import os


def world_size() -> int:
    size = int(os.environ.get("WORLD_SIZE", "1"))
    if size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {size}")
    return size


def per_rank_batch(total: int, world: int | None = None) -> int:
    """Batch each DDP rank should put on its DataLoader."""
    total = int(total)
    world = world_size() if world is None else int(world)
    if world < 1:
        raise ValueError(f"world size must be >= 1, got {world}")
    if total < world or total % world != 0:
        raise ValueError(f"batch {total} is not divisible by {world} GPUs")
    return total // world
