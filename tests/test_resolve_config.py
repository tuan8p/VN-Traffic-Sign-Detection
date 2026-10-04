from pathlib import Path
from vn_tsd.config.resolve import resolve_config

ROOT = Path(__file__).resolve().parents[1]

def test_shared_loads():
    cfg = resolve_config(shared_yaml=ROOT / "configs/shared.yaml")
    assert cfg["seed"] == 42
    assert cfg["project"]["name"] == "VN-Traffic-Sign-Detection"
    assert cfg["data"]["num_classes"] == 52
    assert cfg["online_preprocess"]["enabled"] is False

def test_pipeline_overrides_online_flags_but_stays_off():
    cfg = resolve_config(
        pipeline_yaml=ROOT / "configs/pipelines/yolo26m.yaml",
        shared_yaml=ROOT / "configs/shared.yaml",
        runtime_yaml=ROOT / "configs/runtime/local.yaml",
    )
    assert cfg["pipeline"] == "yolo26m"
    assert cfg["model"]["weights"] == "yolo26m.pt"
    assert cfg["online_preprocess"]["mosaic"] is True
    assert cfg["online_preprocess"]["enabled"] is False
