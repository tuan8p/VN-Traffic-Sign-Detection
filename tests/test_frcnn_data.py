import json
import shutil

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from frcnn_fixtures import make_cache, config
from vn_tsd.pipelines.faster_rcnn.config import validate_config
from vn_tsd.pipelines.faster_rcnn.dataset import load_cache, parse_labels
from vn_tsd.pipelines.faster_rcnn.transforms import OnlineTransform, affine


def test_cache_class_boxes_negatives_and_relocation(tmp_path):
    root = make_cache(tmp_path/"cache")
    cfg = validate_config(config(root))
    sets, identity = load_cache(cfg)
    image, target = sets["train"][0]
    assert image.shape == (3, 640, 640) and image.dtype == torch.float32
    assert target["boxes"].tolist() == [[240, 240, 400, 400]]
    assert target["labels"].tolist() == [1]
    assert sets["test"][1][1]["boxes"].shape == (0, 4)
    gt = sets["train"].coco_ground_truth()
    assert gt["annotations"][0]["category_id"] == 0
    relocated = tmp_path/"elsewhere"
    shutil.copytree(root, relocated)
    cfg["data"]["root"] = str(relocated)
    _, moved = load_cache(cfg, splits=("test",), saved_names=identity["names"])
    assert moved["splits"]["test"]["sha256"] == identity["splits"]["test"]["sha256"]


@pytest.mark.parametrize("line", ["0 .5 .5 0 .1", "52 .5 .5 .1 .1", "0 nan .5 .1 .1", "0 .99 .5 .5 .5",
                                   "0.0 .5 .5 .1 .1", "0 1.000001 .5 .000001 .1", "0 .5 .5 .1"])
def test_invalid_labels_fail(tmp_path, line):
    path = tmp_path/"bad.txt"
    path.write_text(line)
    with pytest.raises(ValueError, match="bad.txt:1"):
        parse_labels(path)


@pytest.mark.parametrize("defect", ["missing", "orphan", "duplicate", "size", "split_copy", "classmap", "meta"])
def test_cache_rejects_contract_defects(tmp_path, defect):
    from PIL import Image
    root = make_cache(tmp_path/"cache")
    if defect == "missing": (root/"labels/train/0.txt").unlink()
    if defect == "orphan": (root/"labels/train/orphan.txt").write_text("")
    if defect == "duplicate": shutil.copy(root/"images/train/0.png", root/"images/train/0.jpg")
    if defect == "size": Image.new("RGB", (32, 32)).save(root/"images/train/0.png")
    if defect == "split_copy": shutil.copy(root/"images/train/0.png", root/"images/val/0.png")
    if defect == "classmap": (root/"data.yaml").write_text("nc: 52\nnames: []\n")
    if defect == "meta": (root/"meta.json").write_text("[]")
    with pytest.raises((ValueError, FileNotFoundError)):
        load_cache(validate_config(config(root)))


def test_rounding_tolerance_does_not_filter_boxes(tmp_path):
    path = tmp_path/"label.txt"
    path.write_text("51 .000000 .5 .000001 .01")
    boxes, labels = parse_labels(path)
    assert labels == [52] and boxes[0][0] == 0 and boxes[0][2] > 0


def test_master_gate_and_train_only_extra_images(tmp_path):
    root = make_cache(tmp_path/"cache")
    cfg = config(root)
    cfg["online_preprocess"].update(enabled=False, hflip=True, mosaic=True, mixup=True)
    cfg = validate_config(cfg)  # raw hflip ignored while master OFF
    sets, _ = load_cache(cfg)
    baseline = sets["train"].raw(0)
    actual = OnlineTransform(cfg)(sets["train"], 0)
    assert torch.equal(actual[0], baseline[0]) and torch.equal(actual[1]["boxes"], baseline[1]["boxes"])
    cfg["online_preprocess"].update(enabled=True)
    with pytest.raises(ValueError, match="hflip"):
        validate_config(cfg)
    cfg["online_preprocess"].update(hflip=False, jitter=0, mixup_prob=1.0)
    transform = OnlineTransform(validate_config(cfg))
    torch.manual_seed(7)
    a, ta = transform(sets["train"], 0)
    torch.manual_seed(7)
    b, tb = transform(sets["train"], 0)
    assert torch.equal(a, b) and torch.equal(ta["boxes"], tb["boxes"])
    assert len(ta["boxes"]) == 8  # 4 mosaic objects + 4 independent mosaic objects
    assert {tuple(b) for b in ta["boxes"].tolist()} == {(120.,120.,200.,200.), (440.,120.,520.,200.),
                                                       (120.,440.,200.,520.), (440.,440.,520.,520.)}
    assert ta["boxes"].min() >= 0 and ta["boxes"].max() <= 640
    assert sets["val"].transform is None and sets["test"].transform is None


def test_affine_pixels_match_box_translation_and_clip():
    image = torch.zeros(3, 640, 640)
    image[:, 100:200, 100:200] = 1
    target = dict(boxes=torch.tensor([[100., 100., 200., 200.], [630., 1., 640., 10.]]), labels=torch.tensor([1, 2]))
    warped, transformed = affine(image, target, 1.0, 20., 10.)
    assert transformed["boxes"].tolist() == [[120, 110, 220, 210]]
    assert transformed["labels"].tolist() == [1]
    assert warped[:, 120:200, 130:200].mean() > 0.999
    assert torch.allclose(warped[:, 0, 0], torch.full((3,), 114/255))


def test_color_only_changes_pixels_and_affine_empty_targets(tmp_path):
    cfg = config(make_cache(tmp_path/"cache"))
    cfg["online_preprocess"].update(enabled=True, color_jitter=True, jitter=0)
    sets, _ = load_cache(validate_config(cfg))
    raw, target = sets["train"].raw(0)
    image, changed = OnlineTransform(cfg)(sets["train"], 0)
    assert not torch.equal(raw, image)
    assert torch.equal(changed["boxes"], target["boxes"])
    empty_im, empty_t = sets["test"].raw(1)
    _, out = affine(empty_im, empty_t, 1.1, -10, 12)
    assert out["boxes"].shape == (0, 4)


def test_affine_scale_uses_image_center():
    image = torch.zeros(3,640,640)
    image[:,300:340,300:340] = 1
    target = dict(boxes=torch.tensor([[300.,300.,340.,340.]]), labels=torch.tensor([1]))
    warped, out = affine(image, target, 2., 0., 0.)
    assert out["boxes"].tolist() == [[280.,280.,360.,360.]]
    assert warped[:,290:350,290:350].mean() > .99


def test_worker_rng_is_seeded_and_dataset_is_spawnable(tmp_path):
    from torch.utils.data import DataLoader
    from vn_tsd.pipelines.faster_rcnn.dataset import collate_detection
    from vn_tsd.pipelines.faster_rcnn.engine import seed_worker
    cfg = config(make_cache(tmp_path/"cache"))
    cfg["online_preprocess"].update(enabled=True, color_jitter=True)
    cfg = validate_config(cfg)
    datasets, _ = load_cache(cfg)
    dataset = datasets["train"]
    dataset.transform = OnlineTransform(cfg)
    def collect():
        loader = DataLoader(dataset,batch_size=2,num_workers=1,collate_fn=collate_detection,
            worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(17))
        return [(im.clone(),t["boxes"].clone()) for images,targets in loader for im,t in zip(images,targets)]
    a, b = collect(), collect()
    assert len(a) == len(b) == 3
    assert all(torch.equal(x,y) and torch.equal(tx,ty) for (x,tx),(y,ty) in zip(a,b))
