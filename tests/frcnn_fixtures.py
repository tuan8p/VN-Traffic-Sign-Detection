"""Synthetic fixtures only; no downloaded data or weights."""
from pathlib import Path
import json

from PIL import Image
import yaml

from vn_tsd.config.resolve import resolve_config
from vn_tsd.pipelines.faster_rcnn.dataset import REPO_ROOT, read_class_names


def make_cache(root, train=3, val=1, test=2):
    root = Path(root)
    names, _ = read_class_names("configs/classes.csv")
    root.mkdir(parents=True, exist_ok=True)
    (root/"data.yaml").write_text(yaml.safe_dump(dict(
        nc=52, names=names, path=str(root),
        train="images/train", val="images/val", test="images/test",
    ), allow_unicode=True), encoding="utf-8")
    (root/"meta.json").write_text(json.dumps(dict(synthetic=True, unknown_field="allowed")), encoding="utf-8")
    for s, count in (("train", train), ("val", val), ("test", test)):
        for kind in ("images", "labels"):
            (root/kind/s).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            color = {"train": 20, "val": 80, "test": 160}[s]+i
            image = Image.new("RGB", (640, 640), (color, 40+i, 100))
            image.save(root/"images"/s/f"{i}.png")
            # Test pool includes one legitimate negative image.
            label = "" if s == "test" and i == 1 else "0 0.5 0.5 0.25 0.25\n"
            (root/"labels"/s/f"{i}.txt").write_text(label, encoding="utf-8")
    return root


def config(root, run_name="fixture"):
    return resolve_config(REPO_ROOT/"configs/pipelines/faster_rcnn.yaml", REPO_ROOT/"configs/shared.yaml",
        REPO_ROOT/"configs/runtime/local.yaml", overrides=dict(
        data=dict(root=str(root)), model=dict(pretrained=False),
        train=dict(epochs=2, batch_size=2, require_wandb=False, run_name=run_name),
        eval=dict(batch_size=2), runtime_cfg=dict(device="cpu", num_workers=0, mixed_precision=False),
        outputs=dict(zip_after_train=False)))


def tiny_model():
    import torch
    from vn_tsd.pipelines.faster_rcnn.evaluate import LOSS_KEYS
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, images, targets=None):
            if self.training:
                return {k: self.weight.square()*(i+1) for i, k in enumerate(LOSS_KEYS)}
            return [dict(boxes=torch.tensor([[240., 240., 400., 400.]], device=im.device),
                         labels=torch.tensor([1], device=im.device), scores=torch.tensor([0.99], device=im.device)) for im in images]
    return Tiny()
