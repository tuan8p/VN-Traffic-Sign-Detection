import copy
import json
import shutil

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("pycocotools")
from frcnn_fixtures import make_cache, config, tiny_model
from vn_tsd.pipelines.faster_rcnn import model as model_module
from vn_tsd.pipelines.faster_rcnn.train import FasterRCNNPipeline
from vn_tsd.pipelines.faster_rcnn.dataset import content_hash


def test_fit_best_artifacts_wandb_once_and_offline_eval(tmp_path, monkeypatch):
    import wandb
    events = []
    class Run:
        def log(self, row, step): events.append(("log", step))
        def finish(self, **kw): events.append(("finish", kw))
    def init(**kw):
        events.append(("init", kw["name"]))
        assert kw["mode"] == "online"
        return Run()
    monkeypatch.setattr(wandb, "setup", lambda *a, **kw: None)
    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.setattr(wandb, "Settings", lambda **kw: kw)
    monkeypatch.setattr(model_module, "build_faster_rcnn", lambda *a, **kw: tiny_model())
    root = make_cache(tmp_path/"cache")
    cfg = config(root)
    cfg["train"]["require_wandb"] = True
    cfg["outputs"]["zip_after_train"] = True
    run = tmp_path/"run"
    pipe = FasterRCNNPipeline(cfg, run)
    metrics = pipe.fit()
    assert metrics["status"] == "completed" and metrics["best_epoch"] == 1  # tied AP does not overwrite
    assert [e[0] for e in events].count("init") == 1
    assert [e[0] for e in events].count("finish") == 1
    assert len([e for e in events if e[0] == "log"]) == 3  # 2 val epochs + final test
    assert metrics["effective"]["amp"] is False
    for name in ("resolved_config.yaml", "history.json", "metrics.json", "test_metrics.json", "run.zip",
                 "figures/loss.png", "figures/test/examples.json", "checkpoints/checkpoint_best.pt"):
        assert (run/name).is_file(), name
    meta = model_module.read_checkpoint(run/"checkpoints/checkpoint_best.pt")["metadata"]
    history = json.loads((run/"history.json").read_text())
    assert meta["best_epoch"] == 1 and history[1]["best"] is False
    assert meta["supports_training_resume"] is False
    # fit loaded epoch-1 checkpoint, not the final epoch's parameters.
    assert len(history) == 2 and history[1]["train_loss"] < history[0]["train_loss"]
    original_config = (run/"resolved_config.yaml").read_bytes()
    relocated = tmp_path/"moved"
    shutil.copytree(root, relocated)
    shutil.rmtree(relocated/"images/train")
    shutil.rmtree(relocated/"labels/train")
    shutil.rmtree(relocated/"images/val")
    shutil.rmtree(relocated/"labels/val")
    effective = copy.deepcopy(pipe.cfg)
    effective["data"]["root"] = str(relocated)
    monkeypatch.chdir(tmp_path)
    evaluated = FasterRCNNPipeline(effective, run).evaluate()
    assert evaluated == metrics["test"]
    assert (run/"resolved_config.yaml").read_bytes() == original_config
    assert [e[0] for e in events].count("init") == 1
    (relocated/"labels/test/0.txt").write_text("0 .4 .4 .2 .2")
    with pytest.raises(ValueError, match="differs"):
        FasterRCNNPipeline(effective, run).evaluate()
    bad = copy.deepcopy(pipe.cfg)
    bad["train"]["lr"] *= 2
    with pytest.raises(ValueError, match="saved training config"):
        FasterRCNNPipeline(bad, run).evaluate()
    with pytest.raises(RuntimeError, match="does not resume"):
        FasterRCNNPipeline(cfg, run).fit()
    metrics_before = (run/"metrics.json").read_bytes()
    with pytest.raises(RuntimeError, match="does not resume"):
        pipe.fit()
    assert (run/"metrics.json").read_bytes() == metrics_before
    assert not (run/"failure.json").exists()


def test_required_wandb_error_aborts_before_epoch(tmp_path, monkeypatch):
    import wandb
    monkeypatch.setattr(model_module, "build_faster_rcnn", lambda *a, **kw: tiny_model())
    monkeypatch.setattr(wandb, "setup", lambda *a, **kw: None)
    monkeypatch.setattr(wandb, "Settings", lambda **kw: kw)
    def fail(**kw): raise RuntimeError("synthetic W&B connectivity failure")
    monkeypatch.setattr(wandb, "init", fail)
    cfg = config(make_cache(tmp_path/"cache"))
    cfg["train"]["require_wandb"] = True
    run = tmp_path/"run"
    with pytest.raises(RuntimeError, match="connectivity"):
        FasterRCNNPipeline(cfg, run).fit()
    assert json.loads((run/"failure.json").read_text())["status"] == "failed"
    assert not (run/"history.json").exists()
    assert not (run/"metrics.json").exists()


def test_required_wandb_offline_mode_is_not_silent_fallback(tmp_path, monkeypatch):
    cfg = config(make_cache(tmp_path/"cache"))
    cfg["train"]["require_wandb"] = True
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(model_module, "build_faster_rcnn", lambda *a, **kw: pytest.fail("weights must not load before tracking preflight"))
    with pytest.raises(RuntimeError, match="offline/disabled"):
        FasterRCNNPipeline(cfg, tmp_path/"run").fit()


@pytest.mark.parametrize("head_name", ["standard", "custom_mlp", "legacy_standard"])
def test_real_model_checkpoint_strict_roundtrip_and_640_inference(tmp_path, monkeypatch, head_name):
    from vn_tsd.pipelines.faster_rcnn.config import validate_config
    from vn_tsd.pipelines.faster_rcnn.dataset import load_cache
    from torchvision.ops.misc import FrozenBatchNorm2d
    monkeypatch.setattr(torch.hub, "download_url_to_file", lambda *a, **kw: pytest.fail("unexpected download"))
    cfg = config(make_cache(tmp_path/"cache"))
    if head_name == "legacy_standard":
        for key in ("head", "head_hidden_dim", "head_dropout"):
            cfg["model"].pop(key)
    else:
        cfg["model"]["head"] = head_name
    cfg = validate_config(cfg)
    model = model_module.build_faster_rcnn(cfg, initialize=False)
    _, identity = load_cache(cfg)
    meta = dict(resolved_config=cfg, config_sha256=content_hash(cfg), dataset=identity,
                architecture=model_module.architecture(cfg))
    roi_features = torch.randn(3, 1024)
    model.eval()
    with torch.no_grad():
        predictor_before = tuple(v.clone() for v in model.roi_heads.box_predictor(roi_features))
    path = tmp_path/"checkpoint.pt"
    model_module.save_checkpoint(path, model, meta)
    # A buffer survives strict state load; eps survives explicit architecture reconstruction.
    first = next(iter(model.state_dict().values())).clone()
    del model
    reconstructed, saved = model_module.load_checkpoint(path, "cpu")
    with torch.no_grad():
        predictor_after = reconstructed.roi_heads.box_predictor(roi_features)
    assert all(torch.equal(a, b) for a, b in zip(predictor_before, predictor_after))
    from vn_tsd.pipelines.faster_rcnn.heads import CustomFastRCNNPredictor
    assert isinstance(reconstructed.roi_heads.box_predictor, CustomFastRCNNPredictor) == (head_name == "custom_mlp")
    if head_name == "legacy_standard":
        assert "predictor" not in saved["architecture"]
    else:
        assert saved["architecture"]["predictor"]["name"] == head_name
    assert all(m.eps == 0 for m in reconstructed.modules() if isinstance(m, FrozenBatchNorm2d))
    assert torch.equal(first, next(iter(reconstructed.state_dict().values())))
    assert all(p.requires_grad for p in reconstructed.parameters())
    with torch.no_grad():
        prediction = reconstructed([torch.zeros(3, 640, 640)])[0]
    assert len(prediction["boxes"]) <= 100
    assert prediction["labels"].min() >= 1 and prediction["labels"].max() <= 52
    # Actual backward at 128, explicitly a reduced-size offline smoke only.
    reconstructed.transform.min_size, reconstructed.transform.max_size = (128,), 128
    reconstructed.train()
    target = dict(boxes=torch.tensor([[30.,30.,90.,90.]]), labels=torch.tensor([1]))
    losses = reconstructed([torch.zeros(3,128,128)], [target])
    sum(losses.values()).backward()
    assert reconstructed.roi_heads.box_predictor.cls_score.weight.grad is not None
    if head_name == "custom_mlp":
        assert reconstructed.roi_heads.box_predictor.projection[0].weight.grad is not None
        assert reconstructed.roi_heads.box_predictor.bbox_pred.weight.grad is not None
    assert all(torch.isfinite(v) for v in losses.values())
    obj = model_module.read_checkpoint(path)
    obj["metadata"]["architecture"]["frozen_bn_eps"] = 1e-5
    torch.save(obj, tmp_path/"corrupt.pt")
    with pytest.raises(ValueError, match="architecture"):
        model_module.read_checkpoint(tmp_path/"corrupt.pt")
    # After the norm check, changing the recorded head must also be rejected.
    if head_name != "legacy_standard":
        obj["metadata"]["architecture"]["frozen_bn_eps"] = 0.0
        obj["metadata"]["architecture"]["predictor"]["name"] = "custom_mlp" if head_name == "standard" else "standard"
        torch.save(obj, tmp_path/"wrong-head.pt")
        with pytest.raises(ValueError, match="architecture"):
            model_module.read_checkpoint(tmp_path/"wrong-head.pt")
