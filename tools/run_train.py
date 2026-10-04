"""Shared train entry. Not owned by one team.

Team pipeline YOLO26m + preprocessing online fills YOLOPipeline.fit. Team pipeline Faster R-CNN + preprocessing online fills FasterRCNNPipeline.fit.
This file only resolves config and the --online / --no-online switch.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from vn_tsd.config.resolve import resolve_config
from vn_tsd.runtime.run_dir import make_run_dir
from vn_tsd.utils.io import load_yaml
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
    if args.run_name:
        cli.setdefault("train", {})["run_name"] = args.run_name
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
        for sub in ("logs", "figures", "checkpoints"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_dir(cfg.get("outputs", {}).get("root", "outputs"), args.pipeline, args.run_name)

    if args.pipeline == "faster_rcnn":
        from vn_tsd.pipelines.faster_rcnn.train import FasterRCNNPipeline as P
    else:
        from vn_tsd.pipelines.yolo.train import YOLOPipeline as P
    metrics = P(cfg, run_dir).fit()
    print("run_dir:", run_dir)
    print("metrics:", metrics)

if __name__ == "__main__":
    main()
