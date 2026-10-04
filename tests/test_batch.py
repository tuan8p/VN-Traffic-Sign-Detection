import pytest
from vn_tsd.runtime.batch import per_rank_batch, world_size

def test_kaggle_split():
    assert per_rank_batch(8, world=2) == 4
    assert per_rank_batch(16, world=2) == 8

def test_single_process_keeps_total():
    assert per_rank_batch(8, world=1) == 8
    assert world_size() == 1

def test_rejects_uneven_split():
    with pytest.raises(ValueError):
        per_rank_batch(8, world=3)
