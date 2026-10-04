from __future__ import annotations
from typing import Any
from vn_tsd.data.preprocess_online import describe_online_preprocess
from vn_tsd.pipelines.base import BasePipeline
from vn_tsd.utils.io import save_json, save_yaml

class YOLOPipeline(BasePipeline):
    name = "yolo26m"

    def fit(self) -> dict[str, Any]:
        save_yaml(self.cfg, self.run_dir / "resolved_config.yaml")
        metrics = {
            "status": "scaffold",
            "pipeline": self.name,
            "weights": self.cfg.get("model", {}).get("weights", "yolo26m.pt"),
            "online": describe_online_preprocess(self.cfg),
            "primary_metric": "mAP50-95",
        }
        save_json(metrics, self.run_dir / "metrics.json")
        # Log each epoch with vn_tsd.runtime.train_log (train_bar, note_loss, log_epoch, log_early_stop). Same line for both pipelines.
        # Team YOLO: pass the shared train and eval totals to Ultralytics. Do not divide them.
        # TODO [Team pipeline YOLO26m + preprocessing online]: model.train(...) with online aug flags mapped onto Ultralytics hyp
        return metrics

    def evaluate(self) -> dict[str, Any]:
        # TODO [Team pipeline YOLO26m + preprocessing online]: model.val on the test split
        return {"status": "scaffold", "pipeline": self.name}
