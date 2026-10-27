from pathlib import Path
from vn_tsd.config.resolve import resolve_config
from vn_tsd.pipelines.yolo.train import YOLOPipeline

ROOT = Path(__file__).resolve().parents[1]

'''Team Faster R-CNN: bỏ riêng FRCNN khỏi test dùng chung vì fit() đã được hiện thực,
không còn trả status="scaffold". Kiểm tra FRCNN nằm trong test_frcnn_pipeline.py;
giữ nguyên các assertion scaffold của YOLO ở checkout này.
'''

def test_yolo_pipeline_fit_scaffold(tmp_path):
    for name, cls in (("yolo26m", YOLOPipeline),):
        cfg = resolve_config(
            pipeline_yaml=ROOT / f"configs/pipelines/{name}.yaml",
            shared_yaml=ROOT / "configs/shared.yaml",
        )
        metrics = cls(cfg, tmp_path / name).fit()
        assert metrics["status"] == "scaffold"
        assert metrics["pipeline"] == name
        assert "OFF" in metrics["online"]
