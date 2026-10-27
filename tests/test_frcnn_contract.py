"""Lightweight contract tests: these also run in the team's .[dev]-only CI."""
import copy
import json
import subprocess
import sys

import pytest

from vn_tsd.config.resolve import resolve_config
from vn_tsd.eval.metrics import summarize
from vn_tsd.pipelines.faster_rcnn.config import validate_config
from vn_tsd.pipelines.faster_rcnn.dataset import REPO_ROOT


def valid():
    return resolve_config(REPO_ROOT/"configs/pipelines/faster_rcnn.yaml", REPO_ROOT/"configs/shared.yaml",
                          REPO_ROOT/"configs/runtime/local.yaml", dict(data=dict(root="fixture")))


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -1, 1.1])
def test_metric_schema_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        summarize({"mAP50": bad})
    assert summarize({}) == dict.fromkeys(("mAP50-95", "mAP50", "mAP75", "AR100"))


def test_pipeline_import_needs_no_torch_wandb_pycoco():
    script = '''
import sys
class Block:
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','torchvision','wandb','pycocotools','ultralytics'}:
   raise AssertionError('heavy dependency imported: '+fullname)
sys.meta_path.insert(0,Block())
from vn_tsd.pipelines.faster_rcnn.train import FasterRCNNPipeline
from tools.run_eval import main
'''
    subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, check=True, timeout=30, capture_output=True)


def test_cli_saved_config_and_branch_dispatch(tmp_path, monkeypatch):
    from tools.run_eval import main
    from vn_tsd.utils.io import save_yaml
    from vn_tsd.pipelines.faster_rcnn.train import FasterRCNNPipeline
    from vn_tsd.pipelines.yolo.train import YOLOPipeline
    cfg = valid()
    cfg["train"]["lr"] = 0.123
    run = tmp_path/"run"
    save_yaml(cfg, run/"resolved_config.yaml")
    before = (run/"resolved_config.yaml").read_bytes()
    observed = []
    def evaluate(self):
        observed.append(copy.deepcopy(self.cfg))
        return summarize({"mAP50": 0.5})
    monkeypatch.setattr(FasterRCNNPipeline, "evaluate", evaluate)
    monkeypatch.chdir(tmp_path)
    main(["--pipeline", "faster_rcnn", "--run-dir", str(run), "--data-root", str(tmp_path/"moved"), "--device", "cpu"])
    assert observed[0]["train"]["lr"] == 0.123
    assert observed[0]["runtime_cfg"]["device"] == "cpu"
    assert before == (run/"resolved_config.yaml").read_bytes()
    assert json.loads((run/"test_metrics.json").read_text())["metrics"]["AR100"] is None
    cfg["pipeline"] = "yolo26m"
    save_yaml(cfg, run/"resolved_config.yaml")
    monkeypatch.setattr(YOLOPipeline, "evaluate", evaluate)
    main(["--pipeline", "yolo26m", "--run-dir", str(run)])
    with pytest.raises(SystemExit):
        main(["--pipeline", "yolo26m", "--run-dir", str(run), "--device", "cpu"])
    with pytest.raises(ValueError, match="pipeline"):
        main(["--pipeline", "faster_rcnn", "--run-dir", str(run)])


def test_notebook_cells_parse_and_scaffold_train_launch(tmp_path, monkeypatch):
    import ast
    from pathlib import Path
    notebook = json.loads((REPO_ROOT/"notebooks/03_train_faster_rcnn.ipynb").read_text(encoding="utf-8"))
    code_cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for source in code_cells:
        ast.parse(source)
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: commands.append((argv, kw)))
    globals_ = dict(PIPELINE="faster_rcnn", RUN_NAME="faster_rcnn_s42_deadbeef",
        OVERRIDES_PATH=tmp_path/"overrides with spaces.yaml",
        PROCESSED_ROOT=str(tmp_path/"cache with spaces"), ONLINE=False)
    training_cell = next(c for c in code_cells if '"tools.run_train"' in c)
    exec(compile(training_cell, "notebook-train-cell", "exec"), globals_)
    argv, kw = commands[0]
    assert "tools.run_train" in argv
    assert argv[argv.index("--pipeline")+1] == "faster_rcnn"
    assert argv[argv.index("--overrides")+1] == str(globals_["OVERRIDES_PATH"])
    assert argv[argv.index("--data-root")+1] == globals_["PROCESSED_ROOT"]
    assert argv[argv.index("--run-name")+1].endswith("deadbeef")
    assert kw["check"] is True
