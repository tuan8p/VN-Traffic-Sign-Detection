from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from vn_tsd.utils.io import load_yaml, save_json


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a detection run"
    )
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=["faster_rcnn", "yolo26m"],
    )
    parser.add_argument("--run-dir", required=True)

    args = parser.parse_args(argv)

    # 1. Load the saved experiment configuration.
    run_dir = Path(args.run_dir).expanduser().resolve()
    config_path = run_dir / "resolved_config.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy cấu hình: {config_path}"
        )

    cfg = load_yaml(config_path)

    if cfg.get("pipeline") != args.pipeline:
        raise ValueError(
            f"Config dùng pipeline {cfg.get('pipeline')!r}, "
            f"CLI chọn {args.pipeline!r}"
        )

    # 2. Select the implementation.
    # Import only the selected pipeline.
    if args.pipeline == "yolo26m":
        from vn_tsd.pipelines.yolo.train import YOLOPipeline

        pipeline = YOLOPipeline(cfg, run_dir)
    else:
        from vn_tsd.pipelines.faster_rcnn.train import (
            FasterRCNNPipeline,
        )

        pipeline = FasterRCNNPipeline(cfg, run_dir)

    # 3. Evaluate through the shared interface.
    metrics = pipeline.evaluate()

    # The Faster R-CNN implementation currently returns a scaffold.
    if metrics.get("status") == "scaffold":
        raise NotImplementedError(
            f"Evaluation cho {args.pipeline} chưa được triển khai."
        )

    # 4. Save evaluation results separately from training metadata.
    output_path = run_dir / "test_metrics.json"
    save_json(
        {
            "pipeline": args.pipeline,
            "split": "test",
            "data_root": cfg.get("data", {}).get("root"),
            "metrics": metrics,
        },
        output_path,
    )

    print("metrics:", metrics)
    print("saved:", output_path)


if __name__ == "__main__":
    main()