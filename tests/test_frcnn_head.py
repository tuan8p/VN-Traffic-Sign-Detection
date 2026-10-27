"""Custom predictor behavior and preservation of pretrained feature layers."""
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from frcnn_fixtures import config
from vn_tsd.pipelines.faster_rcnn.config import validate_config
from vn_tsd.pipelines.faster_rcnn.heads import CustomFastRCNNPredictor
from vn_tsd.pipelines.faster_rcnn.model import build_faster_rcnn


@pytest.mark.parametrize("shape", [(5, 1024), (5, 1024, 1, 1), (0, 1024)])
def test_custom_head_outputs_and_empty_proposals(shape):
    head = CustomFastRCNNPredictor(1024, 53, hidden_dim=64, dropout=0.2)
    scores, boxes = head(torch.randn(shape))
    assert scores.shape == (shape[0], 53)
    assert boxes.shape == (shape[0], 212)
    assert torch.isfinite(scores).all() and torch.isfinite(boxes).all()


def test_custom_head_gradient_and_dropout_modes():
    torch.manual_seed(7)
    head = CustomFastRCNNPredictor(1024, 53, hidden_dim=64, dropout=0.2)
    features = torch.randn(8, 1024, requires_grad=True)
    head.eval()
    first = head(features)
    assert all(torch.equal(a, b) for a, b in zip(first, head(features)))
    head.train()
    a, b = head(features), head(features)
    assert not torch.equal(a[0], b[0])
    (a[0].square().mean()+a[1].square().mean()).backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    for param in head.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()
    with pytest.raises(ValueError, match="expects"):
        head(torch.zeros(2, 1024, 2, 2))


def test_coco_state_load_preserves_features_before_head_replacement(tmp_path, monkeypatch):
    """Synthetic COCO-shaped state only; no real COCO download/parity claim."""
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    cfg = validate_config(config(tmp_path/"unused-cache"))
    baseline_cfg = {**cfg, "model": {**cfg["model"], "head": "standard"}}
    source = build_faster_rcnn(baseline_cfg, initialize=False)
    source.roi_heads.box_predictor = FastRCNNPredictor(1024, 91)
    with torch.no_grad():
        source.backbone.body.conv1.weight.fill_(0.125)
        source.roi_heads.box_head.fc7.weight.fill_(0.25)
    state = source.state_dict()
    monkeypatch.setattr(torch.hub, "download_url_to_file", lambda *a, **kw: pytest.fail("unexpected download"))
    with patch.object(FasterRCNN_ResNet50_FPN_Weights, "get_state_dict", return_value=state) as load:
        model = build_faster_rcnn(cfg, initialize=True)
    assert load.call_count == 1
    assert isinstance(model.roi_heads.box_predictor, CustomFastRCNNPredictor)
    assert torch.equal(model.backbone.body.conv1.weight, state["backbone.body.conv1.weight"])
    assert torch.equal(model.roi_heads.box_head.fc7.weight, state["roi_heads.box_head.fc7.weight"])
    assert model.roi_heads.box_predictor.cls_score.out_features == 53
    assert all(param.requires_grad for param in model.parameters())
    del model
    broken = dict(state)
    broken.pop("backbone.body.conv1.weight")
    with patch.object(FasterRCNN_ResNet50_FPN_Weights, "get_state_dict", return_value=broken):
        with pytest.raises(RuntimeError, match="Missing key"):
            build_faster_rcnn(cfg, initialize=True)
