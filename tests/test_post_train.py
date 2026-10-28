import json
import os
import subprocess
from pathlib import Path

import pytest

from vn_tsd.runtime.post_train import yolo_run_deferred_test
from vn_tsd.utils.io import save_json


def test_yolo_run_deferred_test_skips_without_ddp(tmp_path, monkeypatch):
    cfg = {"pipeline": "yolo26m", "eval": {"auto_eval_test": True}, "outputs": {"zip_after_train": True}}
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert yolo_run_deferred_test(cfg, tmp_path) is False


def test_yolo_run_deferred_test_runs_eval_and_rezip(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_json({"test_status": "deferred"}, run_dir / "metrics.json")
    save_json({"metrics": {"mAP50-95": 0.1}}, run_dir / "test_metrics.json")

    cfg = {"pipeline": "yolo26m", "eval": {"auto_eval_test": True}, "outputs": {"zip_after_train": True}}
    monkeypatch.setenv("WORLD_SIZE", "2")
    calls = []

    def fake_run(cmd, check, env=None):
        calls.append((cmd, env))
        assert check is True
        if env is not None:
            assert "WORLD_SIZE" not in env
            assert "RANK" not in env
            assert "LOCAL_RANK" not in env

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("vn_tsd.runtime.post_train.zip_run_dir", lambda rd: rd / "run.zip")

    assert yolo_run_deferred_test(cfg, run_dir) is True
    assert "tools.run_eval" in calls[0][0]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["test_status"] == "completed"
    assert metrics["test"]["mAP50-95"] == 0.1
