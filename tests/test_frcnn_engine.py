import random
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("pycocotools")
from frcnn_fixtures import config, tiny_model
from vn_tsd.pipelines.faster_rcnn.engine import WarmupCosine, BestTracker, train_epoch
from vn_tsd.pipelines.faster_rcnn.evaluate import validation_loss, LOSS_KEYS, coco_metrics
from vn_tsd.pipelines.faster_rcnn.distributed import Context, eval_indices


def test_scheduler_successful_updates_and_short_smoke():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=0.005)
    cfg = dict(epochs=3, lr=0.005, min_lr=1e-6, warmup_start_factor=0.01)
    s = WarmupCosine(optimizer, cfg, 2)
    assert s.value(0) == pytest.approx(0.00005)
    assert s.value(2) == pytest.approx(0.005)
    assert s.value(6) == pytest.approx(1e-6)
    s.step()
    before = optimizer.param_groups[0]["lr"]
    # A skipped update intentionally does not call step.
    assert optimizer.param_groups[0]["lr"] == before and s.updates == 1
    cfg["epochs"] = 1
    assert WarmupCosine(optimizer, cfg, 2).value(2) == pytest.approx(0.005)


def test_best_zero_tie_patience():
    best = BestTracker(10)
    assert best.update(0.0, 1) == (True, False)
    for epoch in range(2, 11):
        assert best.update(0.0, epoch) == (False, False)
    assert best.update(0.0, 11) == (False, True)
    assert best.epoch == 1


def test_eval_shard_coverage():
    for n in (1, 3, 5):
        shards = [eval_indices(n, rank, 4) for rank in range(4)]
        assert sorted(i for shard in shards for i in shard) == list(range(n))


def test_diagnostic_loss_restores_rng_flags_buffers_and_weights():
    class Diagnostic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(2)
            self.drop = torch.nn.Dropout()
            self.w = torch.nn.Parameter(torch.ones(()))
            self.register_buffer("mutable", torch.zeros(()))
        def forward(self, images, targets):
            self.mutable.add_(1)
            random.random(); np.random.rand(); torch.rand(())
            assert not self.bn.training and not self.drop.training
            return {k: self.w.square() for k in LOSS_KEYS}
    m = Diagnostic().train()
    m.drop.eval()
    initial = {k: v.clone() for k, v in m.state_dict().items()}
    flags = [x.training for x in m.modules()]
    random.seed(9); np.random.seed(9); torch.manual_seed(9)
    py, np_rng, rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    batch = ([torch.zeros(3, 16, 16)], [dict(boxes=torch.empty(0,4), labels=torch.empty(0,dtype=torch.long))])
    values = validation_loss(m, [batch], torch.device("cpu"), 100)
    assert values == [1.,1.,1.,1.,1]
    assert flags == [x.training for x in m.modules()]
    assert all(torch.equal(initial[k], v) for k, v in m.state_dict().items())
    assert py == random.getstate() and torch.equal(rng, torch.get_rng_state())
    assert np.array_equal(np_rng[1], np.random.get_state()[1])


def test_coco_empty_perfect_and_no_gt():
    gt = dict(info={}, images=[dict(id=1, width=640, height=640)], categories=[dict(id=0, name="fixture")],
        annotations=[dict(id=1, image_id=1, category_id=0, bbox=[10,20,100,100], area=10000, iscrowd=0)])
    assert set(coco_metrics(gt, []).values()) == {0.0}
    pred = [dict(image_id=1, category_id=0, bbox=[10,20,100,100], score=0.9)]
    assert all(v == pytest.approx(1) for v in coco_metrics(gt, pred).values())
    gt["annotations"] = []
    assert set(coco_metrics(gt, []).values()) == {None}


def test_nonfinite_loss_and_all_amp_skipped_abort():
    m = tiny_model()
    ctx = Context(dict(device="cpu"))
    optimizer = torch.optim.SGD(m.parameters(), lr=.005)
    schedule = WarmupCosine(optimizer, dict(epochs=1, lr=.005, min_lr=1e-6, warmup_start_factor=.001), 1)
    batch = ([torch.zeros(3,16,16)], [dict(boxes=torch.empty(0,4), labels=torch.empty(0,dtype=torch.long))])
    class SkipScaler:
        scale_value = 8.
        def scale(self, x): return x
        def get_scale(self): return self.scale_value
        def step(self, opt): pass
        def update(self): self.scale_value /= 2
    with pytest.raises(RuntimeError, match="No successful"):
        train_epoch(m, [batch], optimizer, SkipScaler(), schedule, ctx, 1, 1, True)
    m.weight.data.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="Nonfinite training"):
        train_epoch(m, [batch], optimizer, torch.amp.GradScaler("cuda", enabled=False), schedule, ctx, 1, 1, False)
