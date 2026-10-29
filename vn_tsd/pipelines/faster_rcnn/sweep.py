"""Forge CLI agent -> launcher -> torchrun -> rank-zero W&B run.

prepare is offline; register and agent are explicit network operations.
The launcher never initializes W&B. The CLI agent supplies the run ID and
parameter file, inherited by workers; only the pipeline's rank zero calls init.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from vn_tsd.config.resolve import resolve_config
from vn_tsd.runtime.wandb_urls import wandb_api_base_url
from vn_tsd.utils.io import load_yaml, save_yaml
from .artifacts import write_json
from .config import validate_config
from .dataset import REPO_ROOT, content_hash, file_hash

OBJECTIVE = "sweep/best_val_map"
PARAMETERS = ("train.lr", "train.weight_decay", "model.head_dropout", "online_preprocess.jitter")


def runtime_identity():
    packages = {}
    for name in ("torch", "torchvision", "wandb", "pycocotools", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, packages=packages)


def source_identity():
    # Relevant source bytes, including new untracked modules; exclude outputs/secrets.
    paths = [p for folder in ("vn_tsd", "tools", "configs")
             for p in (REPO_ROOT/folder).rglob("*") if p.suffix in {".py", ".yaml", ".csv"}]
    paths += [REPO_ROOT/name for name in ("pyproject.toml", "pixi.lock") if (REPO_ROOT/name).exists()]
    return {p.relative_to(REPO_ROOT).as_posix(): file_hash(p) for p in sorted(paths)}


def apply_parameters(base, values):
    if set(values) != set(PARAMETERS):
        raise ValueError(f"Expected exactly these sweep parameters: {PARAMETERS}")
    cfg = copy.deepcopy(base)
    if cfg["model"].get("head") != "custom_mlp" or not cfg["online_preprocess"]["enabled"]:
        raise ValueError("Dropout/jitter sweep requires custom_mlp and online enabled")
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid sampled parameter {key}")
        section, field = key.split(".")
        cfg[section][field] = value
    return validate_config(cfg)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_campaign(directory):
    directory = Path(directory).resolve()
    obj = _read(directory/"campaign.json")
    cfg = load_yaml(directory/"base_config.yaml")
    spec = load_yaml(directory/"sweep.yaml")
    if content_hash(cfg) != obj["base_config_sha256"] or content_hash(spec) != obj["sweep_sha256"]:
        raise ValueError("Campaign configuration changed; prepare a new campaign")
    if source_identity() != obj["source_files"]:
        raise ValueError("Source changed since prepare; restore pinned source or prepare a new campaign")
    if runtime_identity() != obj["runtime"]:
        raise ValueError("Runtime packages changed since prepare; restore pinned environment")
    return obj, cfg, spec


def prepare(directory, cfg, search, *, run_cap, nproc, test_policy):
    if type(run_cap) is not int or run_cap < 1 or type(nproc) is not int or nproc not in (1, 2):
        raise ValueError("Positive run cap and nproc=1 or 2 required")
    if test_policy not in ("shared", "skip"):
        raise ValueError("Choose test policy explicitly: shared or skip")
    cfg = copy.deepcopy(cfg)
    if test_policy == "skip":
        cfg["eval"]["auto_eval_test"] = False
    cfg["data"]["root"] = str(Path(cfg["data"]["root"]).expanduser().resolve())
    cfg["data"]["class_table"] = str((REPO_ROOT/cfg["data"]["class_table"]).resolve())
    cfg["train"].pop("run_name", None)
    cfg = apply_parameters(cfg, {k: cfg[k.split('.')[0]][k.split('.')[1]] for k in PARAMETERS})
    if not cfg["train"]["require_wandb"] or not cfg["model"]["pretrained"]:
        raise ValueError("Sweep requires W&B and fresh pretrained initialization")
    if cfg["train"]["batch_size"] % nproc or cfg["eval"]["batch_size"] % nproc:
        raise ValueError("Train/eval total batches must divide nproc")
    if set(search) != {"method", "metric", "parameters"}:
        raise ValueError("Search accepts only method/metric/parameters; no pruning or hidden overrides")
    if search["method"] not in ("bayes", "random", "grid") or search["metric"] != dict(name=OBJECTIVE, goal="maximize"):
        raise ValueError("Expected supported search method and best-validation objective")
    if set(search["parameters"]) != set(PARAMETERS):
        raise ValueError("Search space must contain the four supported parameters")
    # Validate all bounds/categorical values with the same model/config validator.
    defaults = {key: cfg[key.split('.')[0]][key.split('.')[1]] for key in PARAMETERS}
    for key, definition in search["parameters"].items():
        if set(definition) == {"value"}:
            candidates = [definition["value"]]
        elif set(definition) == {"values"} and definition["values"]:
            candidates = definition["values"]
        elif (set(definition) == {"distribution", "min", "max"}
              and definition["distribution"] == "log_uniform_values"
              and 0 < definition["min"] <= definition["max"] and search["method"] != "grid"):
            candidates = [definition["min"], definition["max"]]
        else:
            raise ValueError(f"Unsupported or invalid search definition: {key}")
        for value in candidates:
            apply_parameters(cfg, {**defaults, key: value})
    from .dataset import load_cache
    _, dataset = load_cache(cfg)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    spec = copy.deepcopy(search)
    spec.update(run_cap=run_cap, program="vn_tsd.pipelines.faster_rcnn.sweep",
                command=["${env}", sys.executable, "-m", "vn_tsd.pipelines.faster_rcnn.sweep",
                         "trial", "--campaign", str(directory), "--params", "${args_json_file}"])
    save_yaml(cfg, directory/"base_config.yaml")
    save_yaml(spec, directory/"sweep.yaml")
    write_json(dataset, directory/"dataset_identity.json")
    write_json(dict(base_config_sha256=content_hash(cfg), sweep_sha256=content_hash(spec),
                    source_files=source_identity(), runtime=runtime_identity(), dataset_sha256=content_hash(dataset),
                    nproc=nproc, test_policy=test_policy, interpreter=sys.executable), directory/"campaign.json")
    return directory


def tracking_environment(cfg):
    if os.environ.get("WANDB_MODE", "").lower() in {"offline", "disabled", "dryrun"}:
        raise ValueError("Sweeps require online W&B; remove offline/disabled mode")
    return dict(os.environ, WANDB_BASE_URL=wandb_api_base_url(cfg["project"]),
                WANDB_ENTITY=cfg["project"]["wandb_entity"], WANDB_PROJECT=cfg["project"]["wandb_project"],
                WANDB_MODE="online")


def register(directory):
    directory = Path(directory).resolve()
    _, cfg, spec = load_campaign(directory)
    target = directory/"sweep_id.json"
    if target.exists():
        return _read(target)  # Rerunning this command never creates another sweep.
    environment = tracking_environment(cfg)
    # Reserve before network call: an uncertain response must be recovered manually.
    with (directory/"registration_started.json").open("x", encoding="utf-8") as stream:
        json.dump({"status": "registration_started"}, stream)
    os.environ.update({k: v for k, v in environment.items() if k.startswith("WANDB_")})
    import wandb
    wandb.setup(wandb.Settings(base_url=wandb_api_base_url(cfg["project"])))
    project = cfg["project"]
    sweep_id = wandb.sweep(spec, entity=project["wandb_entity"], project=project["wandb_project"])
    result = dict(entity=project["wandb_entity"], project=project["wandb_project"], sweep_id=sweep_id,
                  path=f"{project['wandb_entity']}/{project['wandb_project']}/{sweep_id}")
    write_json(result, target)
    return result


def agent(directory, count):
    _, cfg, _ = load_campaign(directory)
    if type(count) is not int or count < 1:
        raise ValueError("Agent count must be positive and finite")
    identity = _read(Path(directory)/"sweep_id.json")
    # The CLI agent supervises one launcher at a time, including stop signals.
    subprocess.run([sys.executable, "-m", "wandb", "agent", "--count", str(count), identity["path"]],
                   cwd=REPO_ROOT, env=tracking_environment(cfg), check=True)


def verify_trial(cfg, dataset, directory):
    """Checked by all ranks before model initialization; pipeline already checks rank config hashes."""
    trial = _read(Path(directory)/"sweep_trial.json")
    if content_hash(cfg) != trial["config_sha256"] or content_hash(dataset) != trial["dataset_sha256"]:
        raise ValueError("Sweep config/dataset differs from frozen trial")
    return trial


def verify_run(run, cfg, trial):
    if run.id != trial["run_id"] or run.sweep_id != trial["sweep_id"]:
        raise ValueError("W&B run/sweep ID differs from agent assignment")
    sampled = {key: run.config[key] for key in PARAMETERS}
    if sampled != trial["parameters"]:
        raise ValueError("W&B sampled parameters differ from resolved training configuration")
    for key, value in sampled.items():
        section, field = key.split(".")
        if cfg[section][field] != value:
            raise ValueError("Sampled parameter not applied to model/training")


def trial(directory, params_path):
    directory = Path(directory).resolve()
    campaign, base, _ = load_campaign(directory)
    run_id, sweep_id = os.environ.get("WANDB_RUN_ID", ""), os.environ.get("WANDB_SWEEP_ID", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or not re.fullmatch(r"[A-Za-z0-9_-]+", sweep_id):
        raise ValueError("trial must be started by the W&B CLI agent with run/sweep IDs")
    registered = _read(directory/"sweep_id.json")
    if sweep_id != registered["sweep_id"]:
        raise ValueError("Agent sweep ID differs from campaign")
    values = _read(params_path)
    cfg = apply_parameters(base, values)
    cfg["train"]["run_name"] = f"frcnn_sweep_s{cfg['seed']}_{run_id}_{hashlib.sha256(run_id.encode()).hexdigest()[:8]}"
    run_dir = directory/"trials"/run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    save_yaml(cfg, run_dir/"trial_config.yaml")
    write_json(dict(run_id=run_id, sweep_id=sweep_id, parameters=values,
                    config_sha256=content_hash(cfg), dataset_sha256=campaign["dataset_sha256"],
                    campaign=str(directory)), run_dir/"sweep_trial.json")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc_per_node={campaign['nproc']}", "--max_restarts=0", "--module",
               "vn_tsd.pipelines.faster_rcnn.sweep", "worker", "--run-dir", str(run_dir)]
    write_json(dict(status="running", command=command), run_dir/"launch_status.json")
    # Each trial gets its own SDK logs, never the notebook's netrc or secrets.
    env = tracking_environment(cfg)
    env.update(WANDB_DIR=str(run_dir), WANDB_RESUME="never", PYTHONUNBUFFERED="1")
    process = None
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Agent stopped trial")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (run_dir/"console.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                       start_new_session=os.name != "nt")
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
        metrics = _read(run_dir/"metrics.json")
        score = metrics.get("best_val", {}).get("mAP50-95")
        if metrics.get("status") != "completed" or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Worker did not produce a completed trial with a valid objective")
        write_json(dict(status="completed", objective=score, best_epoch=metrics["best_epoch"]), run_dir/"launch_status.json")
    except BaseException as exc:
        write_json(dict(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error_type=type(exc).__name__), run_dir/"launch_status.json")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        if process is not None and process.poll() is None:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=10)
        if process is not None and process.stdout is not None:
            process.stdout.close()
        from vn_tsd.runtime.pack import zip_run_dir
        archive = zip_run_dir(run_dir)  # Refresh after console log/launch receipt are closed.
        (run_dir/"run.zip.sha256").write_text(file_hash(archive)+"  run.zip\n", encoding="utf-8")
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare", help="Offline: freeze source/config/cache and search definition")
    p.add_argument("--campaign", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--search", default="configs/sweeps/faster_rcnn.yaml")
    p.add_argument("--runtime", default="configs/runtime/kaggle.yaml")
    p.add_argument("--overrides")
    p.add_argument("--run-cap", required=True, type=int)
    p.add_argument("--nproc", type=int, default=2)
    p.add_argument("--test-policy", required=True, choices=("shared", "skip"))
    for action in ("register", "agent", "trial"):
        p = sub.add_parser(action)
        p.add_argument("--campaign", required=True)
        if action == "agent":
            p.add_argument("--count", type=int, required=True)
        if action == "trial":
            p.add_argument("--params", required=True)
    p = sub.add_parser("worker")
    p.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        overrides = load_yaml(args.overrides) if args.overrides else {}
        overrides.setdefault("data", {})["root"] = args.data_root
        cfg = resolve_config(REPO_ROOT/"configs/pipelines/faster_rcnn.yaml", REPO_ROOT/"configs/shared.yaml",
                             args.runtime, overrides=overrides)
        print(prepare(args.campaign, cfg, load_yaml(args.search), run_cap=args.run_cap,
                      nproc=args.nproc, test_policy=args.test_policy))
    elif args.action == "register":
        print(register(args.campaign))
    elif args.action == "agent":
        agent(args.campaign, args.count)
    elif args.action == "trial":
        trial(args.campaign, args.params)
    else:
        from .train import FasterRCNNPipeline
        FasterRCNNPipeline(load_yaml(Path(args.run_dir)/"trial_config.yaml"), Path(args.run_dir)).fit()


if __name__ == "__main__":
    main()
