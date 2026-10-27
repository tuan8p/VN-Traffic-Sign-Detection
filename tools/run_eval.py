"""CLI dùng chung: đọc config của run, gọi evaluate(), lưu test_metrics.json.

TODO [Team pipeline Faster R-CNN + preprocessing online]: hiện thực evaluate()
trong pipeline của mình, trả dict bốn metric theo eval/metrics.py. Không thêm
inference/COCOeval ở CLI này. Config đã lưu được dùng nguyên, không merge lại
.env/YAML hiện tại. Nhánh R-CNN vẫn scaffold và sẽ báo chưa triển khai.
Mỗi pipeline trả dict bốn metric theo eval/metrics.py. Inference/COCOeval nằm
trong pipeline, không ở CLI. Config đã lưu được dùng nguyên; nhánh Faster R-CNN
cho relocation data-root/device tường minh và kiểm cùng test content.
"""
from __future__ import annotations

import argparse
import copy
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
    parser.add_argument("--data-root", default=None, help="FRCNN only: relocate identical test cache")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="FRCNN only: explicit evaluation device")

    args = parser.parse_args(argv)

    # 1. Load the saved experiment configuration.
    run_dir = Path(args.run_dir).expanduser().resolve()
    config_path = run_dir / "resolved_config.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy cấu hình: {config_path}"
        )

    cfg = load_yaml(config_path)
    cfg = copy.deepcopy(cfg)
    if args.pipeline != "faster_rcnn" and (args.data_root is not None or args.device is not None):
        parser.error("--data-root/--device currently apply to faster_rcnn only")
    if args.data_root is not None:
        cfg.setdefault("data", {})["root"] = str(Path(args.data_root).expanduser().resolve())
    if args.device is not None:
        cfg.setdefault("runtime_cfg", {})["device"] = args.device

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
    # Never write scaffold output as a measured evaluation.
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