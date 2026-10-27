"""Post-training steps that must run outside DDP (rank 0 only)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from vn_tsd.runtime.pack import zip_run_dir
import json

from vn_tsd.utils.io import save_json


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def yolo_run_deferred_test(cfg: dict[str, Any], run_dir: Path) -> bool:
    """Single-GPU COCO test after YOLO DDP train. Caller must be rank 0."""
    if cfg.get("pipeline") != "yolo26m":
        return False
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False
    if not cfg.get("eval", {}).get("auto_eval_test", True):
        return False

    run_dir = Path(run_dir).resolve()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.run_eval",
            "--pipeline",
            "yolo26m",
            "--run-dir",
            str(run_dir),
        ],
        check=True,
    )

    metrics_path = run_dir / "metrics.json"
    test_path = run_dir / "test_metrics.json"
    if metrics_path.is_file() and test_path.is_file():
        metrics = _load_json(metrics_path)
        test_doc = _load_json(test_path)
        metrics["test"] = test_doc.get("metrics")
        metrics["test_status"] = "completed"
        metrics["test_reason"] = "post_ddp_single_gpu"
        save_json(metrics, metrics_path)

    if cfg.get("outputs", {}).get("zip_after_train", True):
        zip_run_dir(run_dir)
    return True
