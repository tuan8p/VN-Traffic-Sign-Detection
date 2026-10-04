from __future__ import annotations
from typing import Any
from vn_tsd.data.preprocess_online import describe_online_preprocess
from vn_tsd.pipelines.base import BasePipeline
from vn_tsd.utils.io import save_json, save_yaml

class FasterRCNNPipeline(BasePipeline):
    name = "faster_rcnn"

    def fit(self) -> dict[str, Any]:
        save_yaml(self.cfg, self.run_dir / "resolved_config.yaml")
        metrics = {
            "status": "scaffold",
            "pipeline": self.name,
            "backbone": self.cfg.get("model", {}).get("backbone", "resnet50"),
            "online": describe_online_preprocess(self.cfg),
            "primary_metric": "mAP50-95",
        }
        save_json(metrics, self.run_dir / "metrics.json")
        # Log each epoch with vn_tsd.runtime.train_log (train_bar, note_loss, log_epoch, log_early_stop). Same line for both pipelines.
        # Team Faster R-CNN: use per_rank_batch so the shared total is not applied on every GPU. Box predictor is 53, dataset is 52.
        # TODO [Team pipeline Faster R-CNN + preprocessing online]: train loop (torchvision detection), log to outputs/.../checkpoints
        # When online_preprocess.enabled, apply letterbox, normalize, color_jitter, mosaic, mixup. Keep hflip off unless this run turns it on.
        return metrics

    def evaluate(self) -> dict[str, Any]:
        # TODO [Team pipeline Faster R-CNN + preprocessing online]: COCO-style mAP on the held-out split
        return {"status": "scaffold", "pipeline": self.name}
