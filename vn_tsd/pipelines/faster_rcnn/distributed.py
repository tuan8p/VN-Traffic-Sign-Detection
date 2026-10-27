"""Small phase-boundary helpers; unequal evaluation shards never call DDP forward."""
from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


class Context:
    def __init__(self, runtime):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.world < 1 or not 0 <= self.rank < self.world:
            raise ValueError("Invalid launcher rank/world size")
        requested = runtime.get("device", "cuda")
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; use explicit CPU smoke override")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")
        self.owns_group = False
        if self.world > 1:
            if dist.is_initialized():
                if dist.get_world_size() != self.world or dist.get_rank() != self.rank:
                    raise RuntimeError("Existing process group does not match launcher")
            else:
                timeout_s = runtime.get("ddp_timeout_seconds", 300)
                if not isinstance(timeout_s, int) or timeout_s <= 0:
                    timeout_s = 300
                dist.init_process_group(
                    backend="nccl" if requested == "cuda" else "gloo",
                    timeout=datetime.timedelta(seconds=timeout_s),
                )
                self.owns_group = True

    @property
    def main(self):
        return self.rank == 0

    def gather(self, value):
        if self.world == 1:
            return [value]
        result = [None]*self.world
        dist.all_gather_object(result, value)
        return result

    def phase(self, fn):
        """Propagate controlled errors after every rank has finished this phase."""
        value, error = None, None
        try:
            value = fn()
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [e for e in self.gather(error) if e]
        if errors:
            raise RuntimeError("; ".join(errors))
        return value

    def main_call(self, fn):
        value = self.phase(lambda: fn() if self.main else None)
        if self.world == 1:
            return value
        objects = [value]
        dist.broadcast_object_list(objects, src=0)
        return objects[0]

    def sum(self, values):
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.world > 1:
            dist.all_reduce(tensor)
        return tensor.cpu().tolist()

    def sync_buffers(self, model):
        if self.world > 1:
            for buffer in model.buffers():
                dist.broadcast(buffer, src=0)

    def close(self):
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()


def eval_indices(length, rank, world):
    return list(range(rank, length, world))
