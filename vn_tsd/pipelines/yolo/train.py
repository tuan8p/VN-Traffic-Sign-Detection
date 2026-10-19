from __future__ import annotations
from typing import Any
from pathlib import Path
import shutil


from vn_tsd.data.preprocess_online import describe_online_preprocess
from vn_tsd.pipelines.base import BasePipeline
from vn_tsd.utils.io import save_json, save_yaml


from vn_tsd.data.preprocess_online import (
    describe_online_preprocess,
    resolve_online_preprocess,
)
from vn_tsd.pipelines.yolo.model import build_yolo26m
from ultralytics import YOLO


class YOLOPipeline(BasePipeline):
    name = "yolo26m"

    def _build_train_args(self) -> dict[str, Any]:
        cfg = self.cfg
        model_cfg = cfg["model"]
        train_cfg = cfg["train"]
        runtime_cfg = cfg.get("runtime_cfg", {})
        online = resolve_online_preprocess(cfg)

        root = cfg.get("data", {}).get("root")
        if not isinstance(root, str) or not root.strip():
            raise ValueError(
                "data.root chưa được cấu hình. "
                "Hãy truyền --data-root hoặc đặt DATA_ROOT."
            )

        data_yaml = Path(root).expanduser().resolve() / "data.yaml"
        if not data_yaml.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy dataset config: {data_yaml}"
            )

        if online["hflip"]:
            raise ValueError(
                "Không bật hflip cho bộ biển báo có các lớp đối xứng gương."
            )

        return {
            "data": str(data_yaml),
            "epochs": train_cfg["epochs"],
            "batch": train_cfg["batch_size"],
            "imgsz": model_cfg["imgsz"],
            "lr0": train_cfg["lr0"],
            "seed": cfg["seed"],
            "patience": train_cfg["early_stopping_patience"],
            "amp": train_cfg["amp"],
            "workers": runtime_cfg.get("num_workers", 2),

            "project": str(self.run_dir.resolve()),
            "name": "train",
            "exist_ok": True,

            # Không lật biển báo.
            "fliplr": 0.0,
            "flipud": 0.0,

            # Tắt rõ các augmentation khi online OFF.
            "mosaic": 1.0 if online["mosaic"] else 0.0,
            "mixup": 0.1 if online["mixup"] else 0.0,
            "hsv_h": 0.015 if online["color_jitter"] else 0.0,
            "hsv_s": 0.7 if online["color_jitter"] else 0.0,
            "hsv_v": 0.4 if online["color_jitter"] else 0.0,
            "translate": online["jitter"],
            "scale": online["jitter"],
            "degrees": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
        }

    
    def _save_checkpoint(self, model) -> Path:
        """Copy the best checkpoint to the project output."""

        best_path = Path(model.trainer.best)

        if not best_path.is_file():
            raise FileNotFoundError(
                f"Best checkpoint not found: {best_path}"
            )

        checkpoint_path = (
            self.run_dir / "checkpoints" / "checkpoint_best.pt"
        )

        if best_path.resolve() != checkpoint_path.resolve():
            shutil.copy2(best_path, checkpoint_path)

        return checkpoint_path

    def fit(self) -> dict[str, Any]:

        save_yaml(self.cfg, self.run_dir / "resolved_config.yaml")

        train_args = self._build_train_args()

        model = build_yolo26m(self.cfg)

        model.train(**train_args)

        checkpoint = self._save_checkpoint(model)

        test_metrics = self.evaluate()

        metrics = {
            "status": "trained",
            "pipeline": self.name,
            "weights": self.cfg.get("model", {}).get(
                "weights", "yolo26m.pt"
            ),
            "online": describe_online_preprocess(self.cfg),
            "primary_metric": self.cfg.get("train", {}).get("primary_metric", "mAP50-95"),
            "test_metrics": test_metrics
        }   


        save_json(metrics, self.run_dir / "metrics.json")
        # Log each epoch with vn_tsd.runtime.train_log (train_bar, note_loss, log_epoch, log_early_stop). Same line for both pipelines.
        # Team YOLO: pass the shared train and eval totals to Ultralytics. Do not divide them.
        # TODO [Team pipeline YOLO26m + preprocessing online]: model.train(...) with online aug flags mapped onto Ultralytics hyp
        return metrics

    def evaluate(self) -> dict[str, Any]:
        # TODO [Team pipeline YOLO26m + preprocessing online]: model.val on the test split
        cfg = self.cfg

        # 1. Load best checkpoint
        checkpoint = (
            self.run_dir / "checkpoints" / "checkpoint_best.pt"
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}"
            )

        model = YOLO(str(checkpoint), task="detect")

        # 2. Dataset configuration
        data_yaml = (
            Path(cfg["data"]["root"]) / "data.yaml"
        )

        # 3. Evaluate on test split
        results = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=cfg["model"]["imgsz"],
            batch=cfg["eval"]["batch_size"],
            workers=cfg.get("runtime_cfg", {}).get(
                "num_workers", 2
            ),
            project=str(self.run_dir.resolve()),
            name="test",
            exist_ok=True,
        )

        # 4. Extract metrics
        metrics = {
            "mAP50-95": float(results.box.map),
            "mAP50": float(results.box.map50),
            "mAP75": float(results.box.map75),
        }

        return metrics