"""Ablation knobs must change actual arguments, sampling or image geometry."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from ultralytics.engine.results import Boxes

from frcnn_fixtures import make_cache
from vn_tsd.config.resolve import resolve_config
from vn_tsd.pipelines.yolo.config import validate_config
from vn_tsd.pipelines.yolo.train import YOLOPipeline
from vn_tsd.pipelines.yolo.sampling import ClassAwareSampler, image_sampling_weights, class_aware_loader
from vn_tsd.pipelines.faster_rcnn.dataset import load_cache
from vn_tsd.pipelines.yolo.coco_eval import _predictions_coco

ROOT = Path(__file__).resolve().parents[1]


def config(root):
    return resolve_config(ROOT/'configs/pipelines/yolo26m.yaml', ROOT/'configs/shared.yaml',
                          overrides={'data': {'root': str(root)}})


def test_ablation_arguments_and_master_switch(tmp_path):
    cfg = config(make_cache(tmp_path/'data'))
    cfg['train'].update(multi_scale=0.1, cls_pw=0.25, close_mosaic=20)
    cfg['online_preprocess'].update(enabled=True, mosaic=True, mosaic_prob=0.5,
                                    mixup=True, mixup_prob=0.05)
    pipeline = YOLOPipeline(validate_config(cfg), tmp_path/'run')
    args = pipeline._build_train_args()
    assert args['mosaic'] == 0.5 and args['mixup'] == 0.05
    assert args['multi_scale'] == 0.1 and args['cls_pw'] == 0.25
    assert args['close_mosaic'] == 20
    pipeline.cfg['online_preprocess']['enabled'] = False
    args = pipeline._build_train_args()
    assert args['mosaic'] == args['mixup'] == 0


def test_missing_native_class_weight_support_fails(tmp_path):
    cfg = config(make_cache(tmp_path/'data'))
    cfg['train']['cls_pw'] = 0.25
    with patch('ultralytics.cfg.DEFAULT_CFG_DICT', {}):
        with pytest.raises(RuntimeError, match='cls_pw'):
            YOLOPipeline(cfg, tmp_path/'run')._build_train_args()


@pytest.mark.parametrize('section,key,value', [
    ('train', 'multi_scale', -0.1), ('train', 'cls_pw', 1.1),
    ('train', 'class_aware_sampling', {'enabled': 'yes'}),
    ('train', 'class_aware_sampling', {'enabled': True, 'max_weight': 0.5}),
    ('online_preprocess', 'mosaic_prob', 1.1),
    ('online_preprocess', 'mixup_prob', float('nan')),
])
def test_invalid_ablation_value_rejected(section, key, value):
    cfg = config('unused')
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_960_input_uses_existing_640_cache_and_original_box_coordinates(tmp_path):
    root = make_cache(tmp_path/'data', train=1, val=1, test=1)
    cfg = config(root)
    cfg['model']['imgsz'] = 960
    cfg = validate_config(cfg)
    args = YOLOPipeline(cfg, tmp_path/'run')._build_train_args()
    assert args['imgsz'] == 960
    datasets, _ = load_cache(cfg)
    dataset = datasets['test']
    gt = dataset.coco_ground_truth()
    assert gt['images'][0]['width'] == gt['images'][0]['height'] == 640
    assert gt['annotations'][0]['bbox'] == [240, 240, 160, 160]
    # Ultralytics returns xyxy in original image coordinates, even at imgsz=960.
    boxes = Boxes(torch.tensor([[240., 240., 400., 400., .9, 0.]]), orig_shape=(640, 640))
    model = Mock()
    model.predict.return_value = [SimpleNamespace(boxes=boxes)]
    predictions = _predictions_coco(model, dataset, {'score_thresh': .001, 'nms_thresh': .5, 'detections_per_img': 100}, imgsz=960)
    assert model.predict.call_args.kwargs['imgsz'] == 960
    assert predictions[0]['bbox'] == gt['annotations'][0]['bbox']


def labels():
    return [{'cls': np.array([[0]])} for _ in range(15)] + [{'cls': np.array([[1]])}]


def test_sampling_weights_are_mild_and_based_on_image_presence():
    weights = image_sampling_weights(labels(), 2, power=.25, max_weight=2)
    assert weights[:15].tolist() == [1.] * 15
    assert 1 < weights[-1] <= 2
    repeated = image_sampling_weights([{'cls': np.array([[0], [0]])}, {'cls': np.array([[1]])}], 2)
    assert repeated.tolist() == [1., 1.]


def test_weighted_global_stream_is_sharded_deterministically():
    dataset = list(range(16))
    weights = [1.] * 15 + [2.]
    single = ClassAwareSampler(dataset, weights, seed=42)
    a = ClassAwareSampler(dataset, weights, rank=0, world_size=2, seed=42)
    b = ClassAwareSampler(dataset, weights, rank=1, world_size=2, seed=42)
    merged = [i for pair in zip(a, b) for i in pair]
    assert merged == list(single)
    assert len(list(a)) == len(list(b)) == 8
    assert list(a) == list(a)
    a.set_epoch(1)
    assert list(a) != merged[::2]


def test_sampling_increases_rare_image_exposure():
    sampler = ClassAwareSampler(list(range(16)), [1.] * 15 + [2.], seed=42)
    rare = 0
    for epoch in range(500):
        sampler.set_epoch(epoch)
        rare += list(sampler).count(15)
    assert .085 < rare / (500 * 16) < .15  # uniform probability would be .0625


def test_sampling_loader_changes_stream_across_epochs():
    class Dataset(torch.utils.data.Dataset):
        labels = labels()
        def __len__(self): return len(self.labels)
        def __getitem__(self, i): return i
    from ultralytics.data.build import build_dataloader
    native = build_dataloader(Dataset(), batch=4, workers=0, device='cpu')
    loader = class_aware_loader(native, {'power': .25, 'max_weight': 2.}, num_classes=2,
                               rank=0, world_size=1, seed=42)
    first = torch.cat(list(loader)).tolist()
    loader.sampler.set_epoch(1)
    second = torch.cat(list(loader)).tolist()
    assert len(first) == len(second) == 16
    assert first != second


def test_notebook_ablation_cells_compile():
    notebook = json.loads((ROOT/'notebooks/04_train_yolo26m.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(ast.parse(''.join(cell['source'])), '<notebook>', 'exec')


def test_trainer_installs_sampling_only_on_train_loader(monkeypatch):
    from ultralytics.data.build import build_dataloader
    from vn_tsd.pipelines.yolo import trainer as module
    class Dataset(torch.utils.data.Dataset):
        labels = labels()
        def __len__(self): return len(self.labels)
        def __getitem__(self, i): return i
    native = build_dataloader(Dataset(), batch=4, workers=0, device='cpu')
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '1')
    instance = object.__new__(module.LauncherDetectionTrainer)
    instance.sampling_config = {'enabled': True, 'power': .25, 'max_weight': 2.}
    instance.data = {'nc': 2}
    instance.args = SimpleNamespace(seed=42)
    instance.eval_batch_size = 4
    with patch.object(module.DetectionTrainer, 'get_dataloader', return_value=native):
        train = instance.get_dataloader('train', 4, -1, 'train')
        val = instance.get_dataloader('val', 4, -1, 'val')
    assert isinstance(train.sampler, ClassAwareSampler)
    assert val is native


def test_single_gpu_epoch_callback_advances_weighted_sampler():
    from vn_tsd.pipelines.yolo.sampling import set_sampling_epoch
    sampler = ClassAwareSampler(list(range(16)), [1.] * 16, seed=42)
    trainer = SimpleNamespace(epoch=4, train_loader=SimpleNamespace(sampler=sampler))
    set_sampling_epoch(trainer)
    assert sampler.epoch == 4
