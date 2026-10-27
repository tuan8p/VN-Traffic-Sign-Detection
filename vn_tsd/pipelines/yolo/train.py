"""YOLO training và evaluation; CLI dùng chung gọi interface BasePipeline.

Đã tích hợp summarize_yolo(): ba AP native và AR100=None. R-CNN cần dùng
summarize() với kết quả evaluator của mình, không dùng adapter Ultralytics.
TODO [Team pipeline YOLO26m + preprocessing online]: nối callback log/history,
kiểm chứng best checkpoint và early stopping dùng validation mAP50-95.
TODO [Hai team pipeline]: thống nhất COCO evaluator trước khi so sánh cuối cùng;
test chỉ đánh giá model đã chọn, không dùng để tune hay chọn checkpoint.
W&B chạy ở rank 0 trong Python; notebook chỉ chuẩn bị Secret và gọi torchrun.
"""
from __future__ import annotations
from typing import Any, Callable
import os
from pathlib import Path
import shutil


from vn_tsd.eval.metrics import summarize_yolo
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

        world = int(os.environ.get("WORLD_SIZE", "1"))
        if world > 1 and train_cfg["batch_size"] % world:
            raise ValueError("Global train batch must be divisible by WORLD_SIZE")
        device = runtime_cfg.get("device", "cpu")
        if world > 1:
            device = list(range(world))
        elif device == "cuda":
            device = 0
        return {
            "device": device,
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

    def fit(
        self, *, callbacks: dict[str, list[Callable]] | None = None
    ) -> dict[str, Any]:

        from .distributed import Context
        from .tracking import Tracking, remove_native_wandb
        ctx = Context()
        tracking = Tracking(self.cfg, self.run_dir, ctx.main)
        success = False
        try:
            if ctx.world == 1 and not self.cfg["train"].get("run_name"):
                from vn_tsd.runtime.run_dir import with_run_hash
                t = self.cfg["train"]
                suffix = "_online" if self.cfg.get("online_preprocess", {}).get("enabled") else ""
                t["run_name"] = with_run_hash(f"yolo26m_s{self.cfg['seed']}_ep{t['epochs']}_bs{t['batch_size']}{suffix}")
            if ctx.world > 1 and self.cfg.get("eval", {}).get("auto_eval_test", True):
                raise ValueError("DDP: set eval.auto_eval_test=false; run tools.run_eval after torchrun exits")
            train_args = ctx.phase(self._build_train_args)
            def prepare():
                if not ctx.main:
                    return
                for sub in ("logs", "figures", "checkpoints"):
                    (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
                if (self.run_dir / "resolved_config.yaml").exists():
                    raise FileExistsError("Run already exists; choose a fresh run directory")
                save_yaml(self.cfg, self.run_dir / "resolved_config.yaml")
                tracking.start()
            ctx.phase(prepare)
            # Download/load weights on rank zero before other ranks read them.
            model = ctx.phase(lambda: build_yolo26m(self.cfg) if ctx.main else None)
            other = ctx.phase(lambda: build_yolo26m(self.cfg) if not ctx.main else None)
            if not ctx.main:
                model = other
            handlers = {
                "on_pretrain_routine_start": [remove_native_wandb],
                "on_train_epoch_end": [tracking.mark_epoch],
                "on_fit_epoch_end": [tracking.log_epoch],
            }
            for event, functions in (callbacks or {}).items():
                handlers.setdefault(event, []).extend(functions)
            for event, functions in handlers.items():
                for function in functions:
                    model.add_callback(event, function)
            ctx.handoff()
            model.train(**train_args)
            # Wait for every worker before rank zero copies the checkpoint.
            ctx.phase(lambda: None)
            def finish_run():
                if not ctx.main:
                    return {"status": "worker_finished", "pipeline": self.name}
                self._save_checkpoint(model)
                test_metrics = self.evaluate() if self.cfg.get("eval", {}).get("auto_eval_test", True) else None
                metrics = {
                    "status": "trained", "pipeline": self.name,
                    "weights": self.cfg["model"]["weights"],
                    "online": describe_online_preprocess(self.cfg),
                    "primary_metric": self.cfg["metrics"]["primary"],
                    "metric_source": "ultralytics-native", "test_metrics": test_metrics,
                    "world_size": ctx.world,
                }
                save_json(metrics, self.run_dir / "metrics.json")
                if tracking.run is not None and test_metrics is not None:
                    tracking.run.summary.update({f"test/{k}": v for k, v in test_metrics.items() if v is not None})
                if tracking.run is not None:
                    import wandb
                    for filename in ("results.png", "confusion_matrix.png", "confusion_matrix_normalized.png"):
                        path = self.run_dir / "train" / filename
                        if path.is_file():
                            tracking.run.log({f"figures/{path.stem}": wandb.Image(str(path))})
                tracking.finish(True)
                return metrics
            metrics = ctx.phase(finish_run)
            success = True
            return metrics
        finally:
            try:
                tracking.finish(success)
            finally:
                ctx.close()

    def evaluate(self) -> dict[str, Any]:
        # TODO [Team pipeline YOLO26m + preprocessing online]: model.val on the test split
        cfg = self.cfg
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("Standalone evaluation must use one process")

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
            device=0 if cfg.get("runtime_cfg", {}).get("device", "cuda") == "cuda" else cfg.get("runtime_cfg", {}).get("device", 0),
            imgsz=cfg["model"]["imgsz"],
            batch=cfg["eval"]["batch_size"],
            workers=cfg.get("runtime_cfg", {}).get(
                "num_workers", 2
            ),
            project=str(self.run_dir.resolve()),
            name="test",
            exist_ok=True,
        )

        # Native AP is available; COCO AR100 remains unavailable.
        return summarize_yolo(results)
