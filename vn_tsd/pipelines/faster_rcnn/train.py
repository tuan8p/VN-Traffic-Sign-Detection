from __future__ import annotations
import copy
import os
import time
from pathlib import Path
from typing import Any

from vn_tsd.pipelines.base import BasePipeline
from vn_tsd.utils.io import save_yaml


class FasterRCNNPipeline(BasePipeline):
    name = "faster_rcnn"

    def fit(self) -> dict[str, Any]:
        # Keep imports lazy: the scaffold CI intentionally installs only .[dev].
        from .distributed import Context
        self._started, self._wandb_run, self._wandb_finished = False, None, False
        ctx = Context(self.cfg.get("runtime_cfg", {}))
        try:
            return self._fit(ctx)
        except Exception as exc:
            # A controlled failure is reported without marking an unfinished run completed.
            # Hard worker/NCCL failures are handled by torchrun's worker supervision.
            if ctx.main:
                from .artifacts import write_json
                try:
                    failure = dict(status="failed", pipeline=self.name, error=str(exc))
                    write_json(failure, self.run_dir/"failure.json")
                    if (self.run_dir/"metrics.json").exists():
                        write_json(failure, self.run_dir/"metrics.json")
                except OSError:
                    pass
            raise
        finally:
            tracker = getattr(self, "_wandb_run", None)
            try:
                if tracker is not None and not getattr(self, "_wandb_finished", False):
                    self._wandb_finished = True
                    tracker.finish(exit_code=1)
            finally:
                ctx.close()

    def _fit(self, ctx):
        import torch
        from torch.utils.data import DataLoader, DistributedSampler, Subset
        from torch.nn.parallel import DistributedDataParallel
        from vn_tsd.runtime.batch import per_rank_batch
        from vn_tsd.runtime.pack import zip_run_dir
        from vn_tsd.runtime.run_dir import with_run_hash
        from vn_tsd.runtime.train_log import history_row, log_epoch, log_early_stop, wandb_epoch_payload
        from vn_tsd.utils.seed import set_seed
        from .artifacts import write_json, provenance, curves, example_ids, overlays
        from .config import validate_config
        from .dataset import load_cache, content_hash, file_hash, collate_detection
        from .distributed import eval_indices
        from .engine import WarmupCosine, BestTracker, seed_worker, train_epoch
        from .evaluate import distributed_evaluate, validation_loss, LOSS_KEYS, evaluator_metadata
        from .model import build_faster_rcnn, architecture, save_checkpoint, read_checkpoint
        from .transforms import OnlineTransform, describe_effective

        cfg = ctx.phase(lambda: validate_config(self.cfg))
        compare = copy.deepcopy(cfg)
        compare["train"].pop("run_name", None)
        if len(set(ctx.gather(content_hash(compare)))) != 1:
            raise ValueError("Training config mismatch between ranks")
        def identity():
            t = cfg["train"]
            online = "_online" if cfg["online_preprocess"]["enabled"] else ""
            name = with_run_hash(t.get("run_name") or f"faster_rcnn_s{cfg['seed']}_ep{t['epochs']}_bs{t['batch_size']}{online}")
            return name, str(Path(self.run_dir).expanduser().resolve())
        name, path = ctx.main_call(identity)
        cfg["train"]["run_name"] = name
        self.cfg, self.run_dir = cfg, Path(path)
        cfg["data"]["root"] = str(Path(cfg["data"]["root"]).expanduser().resolve())
        def prepare_directory():
            if (self.run_dir/"resolved_config.yaml").exists() or (self.run_dir/"checkpoints/checkpoint_best.pt").exists():
                raise FileExistsError("Run already has training artifacts; v1 does not resume/overwrite runs")
            for sub in ("logs", "figures", "checkpoints"):
                (self.run_dir/sub).mkdir(parents=True, exist_ok=True)
            self._started = True
            print(f"[faster_rcnn] canonical run_dir: {self.run_dir}", flush=True)
        ctx.main_call(prepare_directory)
        ctx.main_call(
            lambda: print(
                "[faster_rcnn] loading dataset (Kaggle Input + 2 ranks can take several minutes)…",
                flush=True,
            )
        )
        sets, dataset_id = ctx.phase(lambda: load_cache(cfg))
        if len(set(ctx.gather(content_hash(dataset_id)))) != 1:
            raise ValueError("Dataset content differs between ranks")
        if sets["val"].identity["annotations"] == 0:
            raise ValueError("Validation has no GT; cannot select checkpoint")
        source = ctx.main_call(provenance)
        runtime = cfg["runtime_cfg"]
        amp = bool(cfg["train"].get("amp", True) and runtime.get("mixed_precision", True) and ctx.device.type == "cuda")
        effective = dict(world_size=ctx.world, device=str(ctx.device), amp=amp, online=describe_effective(cfg),
                         initialization="pretrained COCO_V1" if cfg["model"]["pretrained"] else "random weights; smoke only")
        set_seed(cfg["seed"])
        sets["train"].transform = OnlineTransform(cfg)
        batch = per_rank_batch(cfg["train"]["batch_size"], ctx.world)
        eval_batch = per_rank_batch(cfg["eval"]["batch_size"], ctx.world)
        sampler = DistributedSampler(sets["train"], num_replicas=ctx.world, rank=ctx.rank,
            shuffle=True, seed=cfg["seed"], drop_last=False) if ctx.world > 1 else None
        def loader(dataset, batch_size, *, training=False, subset=None, stream=0):
            generator = torch.Generator().manual_seed(cfg["seed"]+ctx.rank+10000*stream)
            return DataLoader(Subset(dataset, subset) if subset is not None else dataset,
                batch_size=batch_size, sampler=sampler if training else None,
                shuffle=training and sampler is None, drop_last=False,
                num_workers=runtime.get("num_workers", 0),
                pin_memory=runtime.get("pin_memory", False) and ctx.device.type == "cuda",
                collate_fn=collate_detection, worker_init_fn=seed_worker, generator=generator)
        train_loader = ctx.phase(lambda: loader(sets["train"], batch, training=True))
        val_loader = ctx.phase(lambda: loader(sets["val"], eval_batch,
            subset=eval_indices(len(sets["val"]), ctx.rank, ctx.world), stream=1))
        loss_loader = ctx.phase(lambda: loader(sets["val"], eval_batch,
            subset=eval_indices(len(sets["val"]), ctx.rank, ctx.world), stream=2))
        test_loader = ctx.phase(lambda: loader(sets["test"], eval_batch,
            subset=eval_indices(len(sets["test"]), ctx.rank, ctx.world), stream=3))
        sample_count = len(sampler) if sampler is not None else len(sets["train"])
        effective.update(train_padding_samples=sample_count*ctx.world-len(sets["train"]),
                         per_rank_train_batch=batch, per_rank_eval_batch=eval_batch,
                         per_rank_tail_batch=sample_count % batch,
                         small_boxes_by_split={k: v.small_boxes for k, v in sets.items()})
        def begin():
            save_yaml(cfg, self.run_dir/"resolved_config.yaml")
            write_json(dataset_id, self.run_dir/"dataset_identity.json")
            write_json(source, self.run_dir/"source_identity.json")
            write_json(effective, self.run_dir/"effective_config.json")
            if any(v.small_boxes for v in sets.values()):
                print("[faster_rcnn] WARNING: cache contains boxes below offline min_box_px; not filtered. Ask cache owner.", flush=True)
            if not cfg["train"]["require_wandb"]:
                print("[faster_rcnn] W&B explicitly disabled: smoke/development run", flush=True)
                return
            import wandb
            if os.environ.get("WANDB_MODE", "").lower() in {"disabled", "offline", "dryrun"}:
                raise ValueError("W&B required: offline/disabled mode is not accepted")
            from vn_tsd.runtime.wandb_urls import wandb_api_base_url, wandb_run_ui_url

            project = cfg["project"]
            api_base = wandb_api_base_url(project)
            wandb.setup(wandb.Settings(base_url=api_base))
            run = wandb.init(entity=project["wandb_entity"], project=project["wandb_project"],
                name=name, config=dict(resolved=cfg, source=source, effective=effective,
                    dataset_class_hash=dataset_id["class_map_sha256"],
                    dataset_split_hashes={s: d["sha256"] for s, d in dataset_id["splits"].items()}),
                settings=wandb.Settings(base_url=api_base), mode="online")
            self._wandb_run = run
            ui_url = wandb_run_ui_url(project, run.id)
            write_json({"id": run.id, "url": ui_url, "api_base_url": api_base}, self.run_dir / "wandb_run.json")
            print(f"[faster_rcnn] W&B: {ui_url}", flush=True)
        ctx.main_call(begin)
        set_seed(cfg["seed"])
        # Validate/init required tracking before any pretrained download or training.
        # Only rank zero loads pretrained weights; DDP broadcasts its parameters.
        model = ctx.phase(lambda: build_faster_rcnn(cfg, initialize=cfg["model"]["pretrained"] and ctx.main).to(ctx.device))
        if ctx.world > 1:
            wrapped = DistributedDataParallel(model, device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None)
        else:
            wrapped = model
        set_seed(cfg["seed"]+ctx.rank)
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["train"]["lr"], momentum=cfg["train"]["momentum"],
                                     weight_decay=cfg["train"]["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        schedule = WarmupCosine(optimizer, cfg["train"], len(train_loader))
        best = BestTracker(cfg["train"]["early_stopping_patience"])
        history, best_metrics = [], None
        checkpoint = self.run_dir/"checkpoints/checkpoint_best.pt"
        test_ids = example_ids(sets["test"])
        for epoch in range(1, cfg["train"]["epochs"]+1):
            start = time.monotonic()
            if sampler is not None:
                sampler.set_epoch(epoch-1)
            training = train_epoch(wrapped, train_loader, optimizer, scaler, schedule, ctx,
                                   epoch, cfg["train"]["epochs"], amp)
            val_metrics, _ = distributed_evaluate(model, sets["val"], val_loader, ctx)
            local_loss = ctx.phase(lambda: validation_loss(model, loss_loader, ctx.device, cfg["seed"]+ctx.rank+100000))
            val_sums = ctx.sum(local_loss)
            val_components = {k: val_sums[i]/val_sums[-1] for i, k in enumerate(LOSS_KEYS)}
            # Rank zero decides; all ranks receive exactly the same best/stop state.
            improved, stop, state = ctx.main_call(lambda: (*best.update(val_metrics["mAP50-95"], epoch),
                                                          (best.best, best.bad, best.epoch)))
            best.best, best.bad, best.epoch = state
            if improved:
                best_metrics = val_metrics
            row = history_row(
                epoch, training["loss"], sum(val_components.values()), val_metrics, training["lr"],
                is_best=improved, patience_count=best.bad,
            )
            history.append(row)
            def end_epoch():
                if improved:
                    metadata = dict(architecture=architecture(cfg), resolved_config=cfg, config_sha256=content_hash(cfg),
                        dataset=dataset_id, source=source, effective=effective, seed=cfg["seed"],
                        best_epoch=epoch, best_val=val_metrics, evaluator=evaluator_metadata(cfg),
                        supports_training_resume=False)
                    save_checkpoint(checkpoint, model, metadata)
                write_json(history, self.run_dir/"history.json")
                log_epoch(epoch, cfg["train"]["epochs"], training["loss"], row["val_loss"], val_metrics,
                          training["lr"], is_best=improved)
                run = getattr(self, "_wandb_run", None)
                if run is not None:
                    payload = wandb_epoch_payload(row)
                    payload["epoch"] = epoch
                    run.log(payload, step=epoch)
                if stop:
                    log_early_stop(epoch, best.patience)
            ctx.main_call(end_epoch)
            if stop:
                break
        obj = ctx.phase(lambda: read_checkpoint(checkpoint))
        ctx.phase(lambda: model.load_state_dict(obj["state_dict"], strict=True))
        test_metrics, predictions = None, []
        if cfg["eval"].get("auto_eval_test", True):
            test_metrics, predictions = distributed_evaluate(model, sets["test"], test_loader, ctx)
        def finish():
            checkpoint_hash = file_hash(checkpoint)
            metrics = dict(status="completed", pipeline=self.name, run_dir=str(self.run_dir), best_epoch=best.epoch,
                best_val=best_metrics, test=test_metrics, test_status="completed" if test_metrics is not None else "not_run",
                test_reason=None if test_metrics is not None else "eval.auto_eval_test=false",
                evaluator=evaluator_metadata(cfg), checkpoint_sha256=checkpoint_hash,
                config_sha256=content_hash(cfg), dataset_split_hashes={s: d["sha256"] for s, d in dataset_id["splits"].items()},
                seed=cfg["seed"], effective=effective, test_used_for_selection=False, supports_training_resume=False)
            curves(history, self.run_dir/"figures")
            if test_metrics is not None:
                write_json(dict(pipeline=self.name, split="test", data_root=cfg["data"]["root"], metrics=test_metrics),
                           self.run_dir/"test_metrics.json")
                write_json(predictions, self.run_dir/"predictions_test.json")
                overlays(sets["test"], predictions, test_ids, self.run_dir/"figures/test", checkpoint_hash)
            run = getattr(self, "_wandb_run", None)
            if run is not None:
                if test_metrics is not None:
                    run.log({f"test/{k}": v for k, v in test_metrics.items()}, step=len(history)+1)
                self._wandb_finished = True
                run.finish()
            write_json(metrics, self.run_dir/"metrics.json")
            if cfg["outputs"].get("zip_after_train", True):
                zip_run_dir(self.run_dir)
            return metrics
        return ctx.main_call(finish)

    def evaluate(self) -> dict[str, Any]:
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Standalone eval is single-process; training already evaluates distributed")
        from .artifacts import write_json
        from .config import validate_config
        from .dataset import load_cache, file_hash, content_hash, collate_detection
        from .distributed import Context
        from .evaluate import distributed_evaluate, evaluator_metadata
        from .model import load_checkpoint
        from torch.utils.data import DataLoader
        import torch
        cfg = validate_config(self.cfg)
        ctx = Context(cfg["runtime_cfg"])
        try:
            path = Path(self.run_dir)/"checkpoints/checkpoint_best.pt"
            model, meta = load_checkpoint(path, ctx.device)
            original = meta["resolved_config"]
            # Only root/device may differ from the saved training configuration.
            compare = copy.deepcopy(cfg)
            compare["data"]["root"] = original["data"]["root"]
            if "device" in original["runtime_cfg"]:
                compare["runtime_cfg"]["device"] = original["runtime_cfg"]["device"]
            else:
                compare["runtime_cfg"].pop("device", None)
            if content_hash(compare) != meta["config_sha256"]:
                raise ValueError("Evaluation must use saved training config; only root/device relocation is allowed")
            sets, identity = load_cache(cfg, splits=("test",), saved_names=meta["dataset"]["names"])
            if identity["splits"]["test"]["sha256"] != meta["dataset"]["splits"]["test"]["sha256"]:
                raise ValueError("Relocated test dataset differs from saved run")
            generator = torch.Generator().manual_seed(cfg["seed"])
            loader = DataLoader(sets["test"], batch_size=cfg["eval"]["batch_size"], shuffle=False,
                num_workers=cfg["runtime_cfg"].get("num_workers", 0), collate_fn=collate_detection, generator=generator)
            metrics, predictions = distributed_evaluate(model, sets["test"], loader, ctx)
            write_json(dict(pipeline=self.name, split="test", evaluator=evaluator_metadata(cfg),
                checkpoint_sha256=file_hash(path), original_data_root=original["data"]["root"],
                effective_data_root=identity["root"], effective_device=str(ctx.device),
                test_sha256=identity["splits"]["test"]["sha256"]), Path(self.run_dir)/"evaluation_meta.json")
            write_json(predictions, Path(self.run_dir)/"predictions_test.json")
            return metrics
        finally:
            ctx.close()
