from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vn_tsd.pipelines.yolo import trainer


@pytest.mark.parametrize('rank', [0, 1])
def test_setup_reuses_group_without_reinitializing(monkeypatch, rank):
    monkeypatch.setenv('RANK', str(rank))
    monkeypatch.setenv('LOCAL_RANK', str(rank))
    dist = Mock()
    dist.is_initialized.return_value = True
    dist.get_rank.return_value = rank
    dist.get_world_size.return_value = 2
    dist.get_backend.return_value = 'nccl'
    monkeypatch.setattr(trainer, 'dist', dist)
    set_device = Mock()
    monkeypatch.setattr(trainer.torch.cuda, 'set_device', set_device)
    instance = SimpleNamespace(world_size=2)
    trainer.LauncherDetectionTrainer._setup_ddp(instance)
    assert str(instance.device) == f'cuda:{rank}'
    set_device.assert_called_once_with(rank)
    dist.init_process_group.assert_not_called()
    dist.destroy_process_group.assert_not_called()


def test_setup_rejects_missing_group(monkeypatch):
    monkeypatch.setattr(trainer.dist, 'is_initialized', lambda: False)
    with pytest.raises(RuntimeError, match='initialized'):
        trainer.LauncherDetectionTrainer._setup_ddp(SimpleNamespace(world_size=2))
