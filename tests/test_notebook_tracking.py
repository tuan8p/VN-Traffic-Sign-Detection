import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from vn_tsd.pipelines.yolo.tracking import Tracking


def test_script_tracking_logs_epoch_once_and_keeps_missing_ar100(tmp_path):
    tracking = Tracking({}, tmp_path, main=True)
    tracking.run = Mock()
    trainer = SimpleNamespace(
        epoch=0, validator=SimpleNamespace(metrics=SimpleNamespace(
            box=SimpleNamespace(map=0.4, map50=0.7, map75=0.45))),
        tloss=None, label_loss_items=lambda *args, **kwargs: {'train/box_loss': 1.2},
        metrics={'val/box_loss': 1.0}, lr={'pg0': 0.01},
    )
    tracking.mark_epoch(trainer)
    tracking.log_epoch(trainer)
    tracking.log_epoch(trainer)
    trainer.epoch = 1
    tracking.log_epoch(trainer)
    assert tracking.run.log.call_count == 1
    assert tracking.run.log.call_args.args[0]['val/mAP75'] == 0.45
    assert 'val/AR100' not in tracking.run.log.call_args.args[0]
    assert json.loads((tmp_path / 'history.json').read_text())[0]['val/AR100'] is None


def test_worker_does_not_initialize_tracking_or_write_history(tmp_path):
    tracking = Tracking({}, tmp_path, main=False)
    tracking.start()
    tracking.mark_epoch(SimpleNamespace(epoch=0))
    tracking.log_epoch(SimpleNamespace(epoch=0))
    assert not list(tmp_path.iterdir())
    assert tracking.run is None


def test_notebook_is_launcher_not_training_process():
    path = Path(__file__).resolve().parents[1] / 'notebooks/04_train_yolo26m.ipynb'
    n = json.loads(path.read_text())
    source = ''
    for cell in n['cells']:
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])
            compile(ast.parse(code), str(path), 'exec')
            source += code
    assert '--nproc_per_node=2' in source
    assert 'wandb.init(' not in source
    assert '.fit(' not in source


def test_tracking_failure_finishes_run_once(tmp_path, monkeypatch):
    import sys
    sdk = Mock(run=None)
    sdk.init.return_value.id = 'test-id'
    sdk.init.return_value.url = 'https://example.invalid/test-run'
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    monkeypatch.setenv('WANDB_MODE', 'online')
    cfg = {'train': {'require_wandb': True, 'run_name': 'shared_run'},
           'project': {'wandb_entity': 'team', 'wandb_project': 'test',
                       'wandb_base_url': 'https://example.invalid'}}
    tracking = Tracking(cfg, tmp_path, main=True)
    tracking.start()
    tracking.finish(False)
    tracking.finish(False)
    sdk.init.return_value.finish.assert_called_once_with(exit_code=1)


def test_group_is_kept_alive_until_context_close(monkeypatch):
    from vn_tsd.pipelines.yolo import distributed
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '2')
    monkeypatch.setattr(distributed.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(distributed.torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(distributed.torch.cuda, 'set_device', Mock())
    state = {'initialized': False}
    dist = Mock()
    dist.is_initialized.side_effect = lambda: state['initialized']
    dist.init_process_group.side_effect = lambda *args, **kwargs: state.update(initialized=True)
    dist.destroy_process_group.side_effect = lambda: state.update(initialized=False)
    monkeypatch.setattr(distributed, 'dist', dist)
    ctx = distributed.Context()
    assert state['initialized']
    # No handoff/destroy occurs before training.
    dist.destroy_process_group.assert_not_called()
    ctx.close()
    assert not state['initialized']
    assert dist.destroy_process_group.call_count == 1
