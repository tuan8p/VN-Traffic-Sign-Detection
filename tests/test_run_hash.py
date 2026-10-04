from vn_tsd.runtime.run_dir import with_run_hash

def test_with_run_hash_appends_eight_hex():
    name = with_run_hash("yolo26m_s42_ep50_bs8")
    assert name.startswith("yolo26m_s42_ep50_bs8_")
    assert len(name.rsplit("_", 1)[-1]) == 8
    assert with_run_hash(name) == name

def test_online_stays_before_hash():
    name = with_run_hash("faster_rcnn_s42_ep20_bs8_online")
    assert name.startswith("faster_rcnn_s42_ep20_bs8_online_")
