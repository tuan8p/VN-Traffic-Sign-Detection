import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("pycocotools")
from frcnn_fixtures import make_cache


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="CPU Gloo unavailable")
@pytest.mark.parametrize("sweep_mode", [False, True])
def test_two_rank_orchestration_empty_eval_shard_and_main_only_io(tmp_path, sweep_mode):
    make_cache(tmp_path/"cache", train=3, val=1, test=1)
    script = Path(__file__).with_name("frcnn_ddp_worker.py")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    env.pop("WANDB_MODE", None)
    command = [sys.executable, str(script), str(tmp_path.resolve())]
    if sweep_mode:
        command.append("--sweep")
    completed = subprocess.run(command,
                               env=env, capture_output=True, text=True, timeout=100)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    a = json.loads((tmp_path/"result_rank0.json").read_text())
    b = json.loads((tmp_path/"result_rank1.json").read_text())
    assert a == b
    assert not (tmp_path/"candidate_1").exists()
    assert not (tmp_path/"wandb_rank1.txt").exists()
    lines = (tmp_path/"wandb_rank0.txt").read_text().splitlines()
    assert lines.count("init") == 1 and lines.count("finish") == 1
    assert (tmp_path/"candidate_0/checkpoints/checkpoint_best.pt").exists()
    if sweep_mode:
        objective = json.loads((tmp_path/"objective.json").read_text())
        assert objective["sweep/best_val_map"] == a["best_val"]["mAP50-95"]
        assert objective["sweep/best_epoch"] == a["best_epoch"]
