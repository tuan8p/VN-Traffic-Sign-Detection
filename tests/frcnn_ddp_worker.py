"""Two CPU/Gloo ranks for orchestration acceptance; never claims NCCL parity."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def worker(rank, work):
    import torch
    import torch.distributed as dist
    from types import SimpleNamespace
    from frcnn_fixtures import config, tiny_model
    from vn_tsd.pipelines.faster_rcnn.distributed import Context
    from vn_tsd.pipelines.faster_rcnn import model as model_module
    from vn_tsd.pipelines.faster_rcnn.train import FasterRCNNPipeline
    work = Path(work)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(work/"gloo_store").as_uri(), rank=rank, world_size=2)
    try:
        ctx = Context(dict(device="cpu"))
        try:
            ctx.phase(lambda: (_ for _ in ()).throw(ValueError("controlled rank-1 error")) if rank == 1 else None)
        except RuntimeError as exc:
            assert "controlled rank-1 error" in str(exc)
        else:
            raise AssertionError("phase failure did not propagate")
        class Run:
            id = "fixture-ddp-run"
            def log(self, values, step):
                with (work/f"wandb_rank{rank}.txt").open("a") as f: f.write(f"log:{step}\n")
            def finish(self, **kw):
                with (work/f"wandb_rank{rank}.txt").open("a") as f: f.write("finish\n")
        def init(**kw):
            (work/f"wandb_rank{rank}.txt").write_text("init\n")
            return Run()
        sys.modules["wandb"] = SimpleNamespace(init=init, setup=lambda *a: None, Settings=lambda **kw: kw)
        model_module.build_faster_rcnn = lambda *a, **kw: tiny_model()
        cfg = config(work/"cache", run_name=f"rank_{rank}")
        cfg["train"]["require_wandb"] = True
        pipe = FasterRCNNPipeline(cfg, work/f"candidate_{rank}")
        result = pipe.fit()
        assert result["effective"]["train_padding_samples"] == 1
        assert result["best_epoch"] == 1
        assert Path(result["run_dir"]).name == "candidate_0"
        assert result["test"]["mAP50-95"] is not None
        (work/f"result_rank{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(sys.argv[1],), nprocs=2, join=True)
