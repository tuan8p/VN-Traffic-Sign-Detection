"""Sweep boundaries: no network, downloads or real model training."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vn_tsd.config.resolve import resolve_config
from vn_tsd.pipelines.faster_rcnn import sweep
from vn_tsd.utils.io import load_yaml, save_yaml

ROOT = Path(__file__).resolve().parents[1]
VALUES = {"train.lr": 0.00073, "train.weight_decay": 0.000031,
          "model.head_dropout": 0.4, "online_preprocess.jitter": 0.05}


def base():
    return resolve_config(ROOT/"configs/pipelines/faster_rcnn.yaml", ROOT/"configs/shared.yaml",
                          ROOT/"configs/runtime/local.yaml", overrides={"data": {"root": "unused"}})


def campaign(tmp_path, monkeypatch):
    from vn_tsd.pipelines.faster_rcnn import dataset
    monkeypatch.setattr(dataset, "load_cache", lambda cfg: ({}, {"identity": "fixed"}))
    directory = sweep.prepare(tmp_path/"campaign", base(), load_yaml(ROOT/"configs/sweeps/faster_rcnn.yaml"),
                              run_cap=2, nproc=2, test_policy="shared")
    sweep.write_json(dict(sweep_id="sweep123", path="HCMUT_IPCV/BTL/sweep123"), directory/"sweep_id.json")
    monkeypatch.setenv("WANDB_RUN_ID", "trial123")
    monkeypatch.setenv("WANDB_SWEEP_ID", "sweep123")
    monkeypatch.delenv("WANDB_MODE", raising=False)
    params = tmp_path/"params.json"
    params.write_text(json.dumps(VALUES))
    return directory, params


def test_mapping_wins_without_mutating_base_and_rejects_typo():
    original = base()
    before = copy.deepcopy(original)
    cfg = sweep.apply_parameters(original, VALUES)
    assert original == before
    for key, value in VALUES.items():
        section, field = key.split(".")
        assert cfg[section][field] == value
    for bad in ({**VALUES, "train.momentum": .8}, {k: v for k, v in VALUES.items() if k != "train.lr"},
                {**VALUES, "train.lr": float("nan")}, {**VALUES, "model.head_dropout": 1.0}):
        with pytest.raises(ValueError):
            sweep.apply_parameters(original, bad)
    for section, field, value in (("model", "head", "standard"), ("online_preprocess", "enabled", False)):
        bad = copy.deepcopy(original)
        bad[section][field] = value
        with pytest.raises(ValueError):
            sweep.apply_parameters(bad, VALUES)


def test_prepare_is_offline_frozen_and_requires_explicit_budget_policy(tmp_path, monkeypatch):
    directory, _ = campaign(tmp_path, monkeypatch)
    obj, cfg, spec = sweep.load_campaign(directory)
    assert spec["run_cap"] == 2 and "early_terminate" not in spec
    assert spec["metric"] == {"name": sweep.OBJECTIVE, "goal": "maximize"}
    assert "${args_json_file}" in spec["command"]
    assert cfg["eval"]["auto_eval_test"] is True
    spec["run_cap"] = 200
    save_yaml(spec, directory/"sweep.yaml")
    with pytest.raises(ValueError, match="configuration changed"):
        sweep.load_campaign(directory)
    with pytest.raises(ValueError, match="run cap"):
        sweep.prepare(tmp_path/"bad", base(), {}, run_cap=0, nproc=2, test_policy="shared")


def test_registration_is_idempotent_and_agent_has_finite_count(tmp_path, monkeypatch):
    directory, _ = campaign(tmp_path, monkeypatch)
    assert sweep.register(directory)["sweep_id"] == "sweep123"  # no SDK import/init needed
    runner = Mock()
    monkeypatch.setattr(sweep.subprocess, "run", runner)
    sweep.agent(directory, 1)
    args = runner.call_args
    assert args.args[0][-3:] == ["--count", "1", "HCMUT_IPCV/BTL/sweep123"]
    assert args.kwargs["env"]["WANDB_BASE_URL"] == "https://api.forge.coreweave.com"
    with pytest.raises(ValueError, match="finite"):
        sweep.agent(directory, 0)


@pytest.mark.parametrize("code", [0, 1])
def test_trial_launches_shared_config_preserves_failed_evidence_and_never_inits_wandb(tmp_path, monkeypatch, code):
    directory, params = campaign(tmp_path, monkeypatch)
    class Process:
        pid = 9999
        stdout = []
        def __init__(self, command, **kwargs):
            self.stdout = iter(["worker log\n"])
            self.command, self.env = command, kwargs["env"]
            run_dir = Path(command[-1])
            cfg = load_yaml(run_dir/"trial_config.yaml")
            for key, value in VALUES.items():
                section, field = key.split(".")
                assert cfg[section][field] == value
            request = json.loads((run_dir/"sweep_trial.json").read_text())
            assert request["config_sha256"] == sweep.content_hash(cfg)
            assert "--nproc_per_node=2" in command and "--max_restarts=0" in command
            assert self.env["WANDB_RUN_ID"] == "trial123"
            if not code:
                sweep.write_json(dict(status="completed", best_val={"mAP50-95": .4764324239767427},
                                      best_epoch=16, test={"mAP50-95": .99}), run_dir/"metrics.json")
        def wait(self, **kw): return code
        def poll(self): return code
    # An iterator with close emulates the real subprocess pipe.
    import io
    class PipeProcess(Process):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stdout = io.StringIO("worker log\n")
    monkeypatch.setattr(sweep.subprocess, "Popen", PipeProcess)
    if code:
        with pytest.raises(sweep.subprocess.CalledProcessError):
            sweep.trial(directory, params)
    else:
        sweep.trial(directory, params)
    run = directory/"trials/trial123"
    status = json.loads((run/"launch_status.json").read_text())
    assert status["status"] == ("failed" if code else "completed")
    assert status.get("objective") == (None if code else .4764324239767427)
    assert (run/"run.zip.sha256").read_text().startswith(sweep.file_hash(run/"run.zip"))
    before = (run/"launch_status.json").read_bytes()
    with pytest.raises(FileExistsError):
        sweep.trial(directory, params)
    assert (run/"launch_status.json").read_bytes() == before


def test_runtime_rejects_dataset_or_config_drift_and_sdk_mismatch(tmp_path):
    cfg = sweep.apply_parameters(base(), VALUES)
    request = dict(config_sha256=sweep.content_hash(cfg), dataset_sha256=sweep.content_hash({"data": 1}),
                   run_id="r1", sweep_id="s1", parameters=VALUES)
    sweep.write_json(request, tmp_path/"sweep_trial.json")
    assert sweep.verify_trial(cfg, {"data": 1}, tmp_path) == request
    with pytest.raises(ValueError, match="frozen"):
        sweep.verify_trial(cfg, {"data": 2}, tmp_path)
    run = SimpleNamespace(id="r1", sweep_id="s1", config=VALUES.copy())
    sweep.verify_run(run, cfg, request)
    run.config["train.lr"] = .009
    with pytest.raises(ValueError, match="sampled"):
        sweep.verify_run(run, cfg, request)


def test_source_drift_aborts_before_agent_or_gpu(tmp_path, monkeypatch):
    directory, _ = campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(sweep, "source_identity", lambda: {})
    with pytest.raises(ValueError, match="Source changed"):
        sweep.agent(directory, 1)


@pytest.mark.parametrize("mismatch", [False, True])
def test_pipeline_single_run_exact_best_objective_or_fail_before_model(tmp_path, monkeypatch, mismatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("pycocotools")
    import wandb
    from frcnn_fixtures import config, make_cache, tiny_model
    from vn_tsd.pipelines.faster_rcnn.dataset import load_cache
    from vn_tsd.pipelines.faster_rcnn import model
    from vn_tsd.pipelines.faster_rcnn.train import FasterRCNNPipeline
    cfg = config(make_cache(tmp_path/"cache"), run_name="frcnn_fixture_12345678")
    cfg["online_preprocess"]["enabled"] = True
    cfg["train"]["require_wandb"] = True
    cfg["eval"]["auto_eval_test"] = False
    cfg = sweep.apply_parameters(cfg, VALUES)
    _, dataset = load_cache(cfg)
    run_dir = tmp_path/"run"
    sweep.write_json(dict(run_id="trial123", sweep_id="sweep123", parameters=VALUES,
                         config_sha256=sweep.content_hash(cfg), dataset_sha256=sweep.content_hash(dataset)),
                     run_dir/"sweep_trial.json")
    run = Mock(id="trial123", sweep_id="sweep123", config=dict(VALUES))
    if mismatch:
        run.config["train.lr"] = .009
    monkeypatch.setenv("WANDB_SWEEP_ID", "sweep123")
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(wandb, "setup", Mock())
    init = Mock(return_value=run)
    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.setattr(wandb, "Settings", lambda **kw: kw)
    builder = Mock(side_effect=lambda *a, **kw: tiny_model())
    monkeypatch.setattr(model, "build_faster_rcnn", builder)
    if mismatch:
        with pytest.raises(RuntimeError, match="sampled parameters"):
            FasterRCNNPipeline(cfg, run_dir).fit()
        builder.assert_not_called()
        run.finish.assert_called_once_with(exit_code=1)
        assert not run.log.called
    else:
        result = FasterRCNNPipeline(cfg, run_dir).fit()
        init.assert_called_once()
        run.finish.assert_called_once_with()
        objective_logs = [call.args[0] for call in run.log.call_args_list if sweep.OBJECTIVE in call.args[0]]
        assert len(objective_logs) == 1
        assert objective_logs[0][sweep.OBJECTIVE] == result["best_val"]["mAP50-95"]
        assert result["test_status"] == "not_run" and result["test"] is None
        assert result["best_epoch"] == 1  # Tie keeps first best, not last epoch.


def test_register_creates_one_sweep_without_a_run_and_recovers_existing_id(tmp_path, monkeypatch):
    directory, _ = campaign(tmp_path, monkeypatch)
    (directory/"sweep_id.json").unlink()
    sdk = Mock()
    sdk.sweep.return_value = "sweep123"
    monkeypatch.setitem(__import__('sys').modules, "wandb", sdk)
    first = sweep.register(directory)
    assert sweep.register(directory) == first
    sdk.sweep.assert_called_once()
    sdk.init.assert_not_called()


def test_uncertain_registration_requires_recovery_not_duplicate_sweep(tmp_path, monkeypatch):
    directory, _ = campaign(tmp_path, monkeypatch)
    (directory/"sweep_id.json").unlink()
    (directory/"registration_started.json").write_text('{}')
    with pytest.raises(FileExistsError):
        sweep.register(directory)


def test_stop_signal_terminates_trial_and_keeps_interrupted_receipt(tmp_path, monkeypatch):
    directory, params = campaign(tmp_path, monkeypatch)
    class InterruptedPipe:
        def __iter__(self):
            raise KeyboardInterrupt("synthetic stop")
        def close(self): pass
    child = Mock(pid=99999, stdout=InterruptedPipe())
    child.poll.return_value = None
    monkeypatch.setattr(sweep.subprocess, "Popen", lambda *a, **kw: child)
    kill_group = Mock()
    monkeypatch.setattr(sweep.os, "killpg", kill_group, raising=False)
    with pytest.raises(KeyboardInterrupt):
        sweep.trial(directory, params)
    if sweep.os.name == "nt":
        child.terminate.assert_called_once()
    else:
        kill_group.assert_called_once_with(child.pid, sweep.signal.SIGTERM)
    status = json.loads((directory/"trials/trial123/launch_status.json").read_text())
    assert status == {"status": "interrupted", "error_type": "KeyboardInterrupt"}


def test_notebook_preflight_defaults_are_ready_for_sandbox():
    import ast
    notebook = json.loads((ROOT/"notebooks/frcnn/frcnn_forge_sweeps.ipynb").read_text(encoding="utf-8"))
    assigned = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == [] and cell["execution_count"] is None
        code = "".join(cell["source"])
        ast.parse(code)
        for node in ast.parse(code).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                try:
                    assigned[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    assert assigned["REGISTER_SWEEP"] is True and assigned["START_AGENT"] is True
    assert assigned["AGENT_COUNT"] == 1 and assigned["STAGE"] == "preflight"
    assert assigned["WANDB_PROJECT"] == "sandbox" and assigned["TEST_POLICY"] == "shared"
    assert assigned["CAMPAIGN_NAME"] == "frcnn_preflight_s42_sandbox_v1"
    assert len(assigned["REPO_REF"]) == 40
    source = ''.join(notebook['cells'][5]['source'])
    assert '"wandb_project": WANDB_PROJECT' in source


def test_notebook_export_puts_one_bundle_and_checksum_at_output_root(tmp_path, monkeypatch):
    import hashlib
    import zipfile
    notebook = json.loads((ROOT/"notebooks/frcnn/frcnn_forge_sweeps.ipynb").read_text(encoding="utf-8"))
    campaign = tmp_path/'campaign'
    run = campaign/'trials/r1'
    run.mkdir(parents=True)
    (campaign/'sweep_id.json').write_text('{"sweep_id":"s1"}')
    (run/'checkpoint.pt').write_bytes(b'fixture weights')
    (run/'run.zip').write_bytes(b'excluded duplicate archive')
    (run/'run.zip.sha256').write_text('old checksum')
    code = ''.join(notebook['cells'][11]['source'])
    # Execute the actual export cell, replacing only the Kaggle output mount.
    code = code.replace('Path("/kaggle/working")', 'OUTPUT_ROOT')
    namespace = dict(CAMPAIGN=campaign, Path=Path, json=json, OUTPUT_ROOT=tmp_path)
    exec(compile(code, 'notebook-export', 'exec'), namespace)
    bundle, checksum = namespace['bundle'], namespace['checksum']
    assert bundle.parent == checksum.parent == tmp_path
    assert checksum.read_text().split()[0] == hashlib.sha256(bundle.read_bytes()).hexdigest()
    with zipfile.ZipFile(bundle) as archive:
        assert archive.read('campaign/trials/r1/checkpoint.pt') == b'fixture weights'
        assert 'campaign/sweep_id.json' in archive.namelist()
        assert not any(p.endswith(('.zip', '.sha256')) for p in archive.namelist())


def test_prepare_real_synthetic_cache_freezes_identity_and_explicit_test_policy(tmp_path):
    from frcnn_fixtures import make_cache
    from vn_tsd.pipelines.faster_rcnn.dataset import load_cache
    cfg = base()
    cfg["data"]["root"] = str(make_cache(tmp_path/"cache"))
    directory = sweep.prepare(tmp_path/"campaign", cfg, load_yaml(ROOT/"configs/sweeps/faster_rcnn.yaml"),
                              run_cap=1, nproc=2, test_policy="skip")
    frozen, resolved, _ = sweep.load_campaign(directory)
    _, dataset = load_cache(resolved)
    assert sweep.content_hash(dataset) == frozen["dataset_sha256"]
    assert resolved["eval"]["auto_eval_test"] is False
    assert cfg["eval"]["auto_eval_test"] is True
