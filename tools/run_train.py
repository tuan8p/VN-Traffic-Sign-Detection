"""Shared train entry. Not owned by one team.

Team pipeline YOLO26m + preprocessing online fills YOLOPipeline.fit. Team pipeline Faster R-CNN + preprocessing online fills FasterRCNNPipeline.fit.
This file only resolves config and the --online / --no-online switch.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from vn_tsd.config.resolve import resolve_config
from vn_tsd.runtime.run_dir import make_run_dir, with_run_hash
from vn_tsd.utils.io import load_yaml, save_json
from vn_tsd.utils.seed import set_seed


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for key, value in b.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train one detection pipeline")
    p.add_argument("--pipeline", required=True, choices=["faster_rcnn", "yolo26m"])
    p.add_argument("--config", default=None)
    p.add_argument("--shared", default="configs/shared.yaml")
    p.add_argument("--runtime", default="configs/runtime/local.yaml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--overrides", default=None, help="YAML merged after the pipeline config. CLI flags win.")
    p.add_argument("--online", action="store_true", help="Force online_preprocess.enabled=true")
    p.add_argument("--no-online", action="store_true", help="Force online_preprocess.enabled=false")
    args = p.parse_args(argv)

    overrides: dict[str, Any] = load_yaml(args.overrides) if args.overrides else {}
    cli: dict[str, Any] = {}
    if args.data_root:
        cli.setdefault("data", {})["root"] = args.data_root
    if args.online and args.no_online:
        p.error("use only one of --online / --no-online")
    if args.online:
        cli.setdefault("online_preprocess", {})["enabled"] = True
    if args.no_online:
        cli.setdefault("online_preprocess", {})["enabled"] = False
    yolo_ddp = args.pipeline == "yolo26m" and int(os.environ.get("WORLD_SIZE", "1")) > 1
    if yolo_ddp and (not args.run_dir or not args.run_name):
        p.error("YOLO DDP requires a shared --run-dir and --run-name from the launcher")
    run_name = args.run_name if yolo_ddp else (with_run_hash(args.run_name) if args.run_name else None)
    if run_name:
        cli.setdefault("train", {})["run_name"] = run_name
    overrides = _merge(overrides, cli)

    pipe_yaml = args.config or f"configs/pipelines/{args.pipeline}.yaml"
    cfg = resolve_config(
        pipeline_yaml=pipe_yaml,
        shared_yaml=args.shared,
        runtime_yaml=args.runtime,
        overrides=overrides or None,
    )
    set_seed(int(cfg.get("seed", 42)))
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not yolo_ddp or int(os.environ.get("RANK", "0")) == 0:
            for sub in ("logs", "figures", "checkpoints"):
                (run_dir / sub).mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(cfg.get("outputs", {}).get("root", "outputs"), args.pipeline, run_name)

    if args.pipeline == "faster_rcnn":
        from vn_tsd.pipelines.faster_rcnn.train import FasterRCNNPipeline as P
    else:
        from vn_tsd.pipelines.yolo.train import YOLOPipeline as P
    # FRCNN rejects reuse of a run directory. Its completed evidence must also
    # survive this outer CLI error handler; YOLO retains its existing policy.
    preserve_frcnn_run = args.pipeline == "faster_rcnn" and any(
        (run_dir / name).exists()
        for name in ("resolved_config.yaml", "metrics.json", "checkpoints/checkpoint_best.pt")
    )
    try:
        metrics = P(cfg, run_dir).fit()
    except Exception as exc:
        if int(os.environ.get("RANK", "0")) == 0 and not preserve_frcnn_run:
            failure = dict(status="failed", pipeline=args.pipeline, error=str(exc))
            try:
                save_json(failure, run_dir / "failure.json")
                save_json(failure, run_dir / "metrics.json")
            except OSError:
                pass
        raise
    if int(os.environ.get("RANK", "0")) == 0:
        if args.pipeline == "yolo26m":
            from vn_tsd.runtime.post_train import yolo_run_deferred_test

            if yolo_run_deferred_test(cfg, run_dir):
                import json

                metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        print("run_dir:", run_dir)
        print("metrics:", metrics)

if __name__ == "__main__":
    main()
