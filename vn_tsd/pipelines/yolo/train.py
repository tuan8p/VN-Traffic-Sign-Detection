"""YOLO training và evaluation; CLI dùng chung gọi interface BasePipeline."""
from __future__ import annotations
from typing import Any, Callable
import os
from pathlib import Path
import shutil

from vn_tsd.data.dataset_contract import data_yaml_for_ultralytics, validate_processed_dataset
from vn_tsd.data.preprocess_online import describe_online_preprocess, resolve_online_preprocess
from vn_tsd.pipelines.base import BasePipeline
from vn_tsd.utils.io import save_json, save_yaml


class YOLOPipeline(BasePipeline):
    name = "yolo26m"

    def _build_train_args(self) -> dict[str, Any]:
        cfg = self.cfg
        model_cfg = cfg["model"]
        train_cfg = cfg["train"]
        runtime_cfg = cfg.get("runtime_cfg", {})
        online = resolve_online_preprocess(cfg)

        root = validate_processed_dataset(cfg)
        data_yaml = data_yaml_for_ultralytics(root, self.run_dir / "ultralytics_data")

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
            "momentum": train_cfg["momentum"],
            "weight_decay": train_cfg["weight_decay"],
            "lrf": train_cfg["lrf"],
            "warmup_epochs": train_cfg["warmup_epochs"],
            "cos_lr": train_cfg["cos_lr"],
            "close_mosaic": train_cfg["close_mosaic"],
            "seed": cfg["seed"],
            # Early stopping by val mAP50-95 is enforced in Tracking, not Ultralytics fitness.
            "patience": 0,
            "eval_batch_size": cfg["eval"]["batch_size"],
            "amp": train_cfg["amp"],
            "workers": runtime_cfg.get("num_workers", 2),

            "project": str(self.run_dir.resolve()),
            "name": "train",
            "exist_ok": True,

            "fliplr": 0.0,
            "flipud": 0.0,

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
        checkpoint_path = self.run_dir / "checkpoints" / "checkpoint_best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Best checkpoint not found (expected val mAP50-95 selection): {checkpoint_path}"
            )
        return checkpoint_path

    def _copy_figures(self) -> None:
        if not self.run_dir.joinpath("figures").exists():
            (self.run_dir / "figures").mkdir(parents=True, exist_ok=True)
        for filename in ("results.png", "confusion_matrix.png", "confusion_matrix_normalized.png"):
            source = self.run_dir / "train" / filename
            if source.is_file():
                shutil.copy2(source, self.run_dir / "figures" / filename)

    def fit(
        self, *, callbacks: dict[str, list[Callable]] | None = None
    ) -> dict[str, Any]:
        from vn_tsd.eval.coco_eval import evaluator_metadata
        from vn_tsd.pipelines.faster_rcnn.dataset import content_hash, file_hash
        from vn_tsd.pipelines.yolo.config import validate_config
        from vn_tsd.pipelines.yolo.model import build_yolo26m
        from vn_tsd.runtime.pack import zip_run_dir
        from .distributed import Context
        from .tracking import Tracking, remove_native_wandb

        self.cfg = validate_config(self.cfg)
        ctx = Context(self.cfg.get("runtime_cfg", {}))
        tracking = Tracking(self.cfg, self.run_dir, ctx.main)
        success = False
        train_args: dict[str, Any] = {}
        try:
            if ctx.world == 1 and not self.cfg["train"].get("run_name"):
                from vn_tsd.runtime.run_dir import with_run_hash
                t = self.cfg["train"]
                suffix = "_online" if self.cfg.get("online_preprocess", {}).get("enabled") else ""
                t["run_name"] = with_run_hash(
                    f"yolo26m_s{self.cfg['seed']}_ep{t['epochs']}_bs{t['batch_size']}{suffix}"
                )
            defer_test = (
                ctx.world > 1 and self.cfg.get("eval", {}).get("auto_eval_test", True)
            )
            train_args = ctx.phase(self._build_train_args)
            effective = dict(
                world_size=ctx.world,
                device=train_args.get("device"),
                amp=bool(self.cfg["train"].get("amp", True)),
                online=describe_online_preprocess(self.cfg),
                metric_source="pycocotools.COCOeval bbox",
            )

            def prepare():
                if not ctx.main:
                    return
                for sub in ("logs", "figures", "checkpoints"):
                    (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
                if (self.run_dir / "resolved_config.yaml").exists():
                    raise FileExistsError("Run already exists; choose a fresh run directory")
                save_yaml(self.cfg, self.run_dir / "resolved_config.yaml")
                save_json(effective, self.run_dir / "effective_config.json")
                tracking.start()

            ctx.phase(prepare)
            model = ctx.phase(lambda: build_yolo26m(self.cfg) if ctx.main else None)
            other = ctx.phase(lambda: build_yolo26m(self.cfg) if not ctx.main else None)
            if not ctx.main:
                model = other
            handlers = {
                "on_pretrain_routine_start": [remove_native_wandb],
                "on_train_epoch_end": [tracking.mark_epoch],
                "on_fit_epoch_end": [tracking.log_epoch],
                "on_train_end": [tracking.on_train_end],
            }
            for event, functions in (callbacks or {}).items():
                handlers.setdefault(event, []).extend(functions)
            for event, functions in handlers.items():
                for function in functions:
                    model.add_callback(event, function)
            if ctx.world > 1:
                from .trainer import LauncherDetectionTrainer
                model.train(trainer=LauncherDetectionTrainer, **train_args)
            else:
                model.train(**train_args)
            ctx.phase(lambda: None)

            def finish_run():
                if not ctx.main:
                    return {"status": "worker_finished", "pipeline": self.name}
                checkpoint = self._save_checkpoint(model)
                test_metrics = None
                test_payload = None
                if self.cfg.get("eval", {}).get("auto_eval_test", True) and not defer_test:
                    test_payload = self.evaluate(detailed=True)
                    test_metrics = test_payload["metrics"]
                self._copy_figures()
                best_epoch = tracking.best_epoch
                if best_epoch is None:
                    raise RuntimeError("Training finished without a val mAP50-95 best checkpoint")
                metrics = dict(
                    status="completed",
                    pipeline=self.name,
                    run_dir=str(self.run_dir),
                    best_epoch=best_epoch,
                    best_val=tracking.best_val,
                    test=test_metrics,
                    test_status=(
                        "deferred"
                        if defer_test
                        else ("completed" if test_metrics is not None else "not_run")
                    ),
                    test_reason=(
                        "post_ddp_single_gpu"
                        if defer_test
                        else (None if test_metrics is not None else "eval.auto_eval_test=false")
                    ),
                    metric_source="pycocotools.COCOeval bbox",
                    evaluator=evaluator_metadata(self.cfg),
                    checkpoint_sha256=file_hash(checkpoint),
                    config_sha256=content_hash(self.cfg),
                    seed=self.cfg["seed"],
                    effective=effective,
                    test_used_for_selection=False,
                    supports_training_resume=False,
                )
                save_json(metrics, self.run_dir / "metrics.json")
                if test_payload is not None:
                    root = self.cfg.get("data", {}).get("root")
                    save_json(
                        dict(
                            pipeline=self.name,
                            split="test",
                            data_root=root,
                            metrics=test_payload["metrics"],
                            metrics_native=test_payload.get("metrics_native"),
                            evaluator=test_payload["evaluator"],
                        ),
                        self.run_dir / "test_metrics.json",
                    )
                    save_json(
                        test_payload["predictions"],
                        self.run_dir / "predictions_test.json",
                    )
                if tracking.run is not None and test_metrics is not None:
                    tracking.run.summary.update(
                        {f"test/{k}": v for k, v in test_metrics.items() if v is not None}
                    )
                if tracking.run is not None:
                    import wandb
                    for filename in ("results.png", "confusion_matrix.png", "confusion_matrix_normalized.png"):
                        path = self.run_dir / "figures" / filename
                        if path.is_file():
                            tracking.run.log({f"figures/{path.stem}": wandb.Image(str(path))})
                tracking.finish(True)
                if self.cfg.get("outputs", {}).get("zip_after_train", True) and not defer_test:
                    zip_run_dir(self.run_dir)
                return metrics

            metrics = ctx.phase(finish_run)
            success = True
            return metrics
        except Exception as exc:
            if ctx.main:
                failure = dict(status="failed", pipeline=self.name, error=str(exc))
                try:
                    save_json(failure, self.run_dir / "failure.json")
                    save_json(failure, self.run_dir / "metrics.json")
                except OSError:
                    pass
            raise
        finally:
            try:
                tracking.finish(success)
            finally:
                ctx.close()

    def evaluate(self, *, detailed: bool = False) -> dict[str, Any]:
        from vn_tsd.pipelines.yolo.config import validate_config
        from vn_tsd.pipelines.yolo.coco_eval import evaluate_split_coco, evaluate_split_native

        cfg = validate_config(self.cfg)
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("Standalone evaluation must use one process")
        validate_processed_dataset(cfg)

        checkpoint = self.run_dir / "checkpoints" / "checkpoint_best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}"
            )

        coco = evaluate_split_coco(cfg, checkpoint, "test")
        if not detailed:
            return coco["metrics"]
        native = evaluate_split_native(cfg, checkpoint, "test", self.run_dir)
        return {**coco, "metrics_native": native}
