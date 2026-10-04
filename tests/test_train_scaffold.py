from pathlib import Path
from vn_tsd.config.resolve import resolve_config
from vn_tsd.pipelines.faster_rcnn.train import FasterRCNNPipeline
from vn_tsd.pipelines.yolo.train import YOLOPipeline

ROOT = Path(__file__).resolve().parents[1]

def test_both_pipelines_fit_scaffold(tmp_path):
    for name, cls in (("faster_rcnn", FasterRCNNPipeline), ("yolo26m", YOLOPipeline)):
        cfg = resolve_config(
            pipeline_yaml=ROOT / f"configs/pipelines/{name}.yaml",
            shared_yaml=ROOT / "configs/shared.yaml",
        )
        metrics = cls(cfg, tmp_path / name).fit()
        assert metrics["status"] == "scaffold"
        assert metrics["pipeline"] == name
        assert "OFF" in metrics["online"]
