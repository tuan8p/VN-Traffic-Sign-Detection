from vn_tsd.data.preprocess_online import resolve_online_preprocess

def test_master_switch_off_ignores_ops():
    flags = resolve_online_preprocess({
        "online_preprocess": {"enabled": False, "mosaic": True, "hflip": True}
    })
    assert flags["enabled"] is False
    assert flags["mosaic"] is False
    assert flags["hflip"] is False

def test_master_switch_on_keeps_ops():
    flags = resolve_online_preprocess({
        "online_preprocess": {"enabled": True, "mosaic": True, "hflip": False, "jitter": 0.2}
    })
    assert flags["enabled"] is True
    assert flags["mosaic"] is True
    assert flags["hflip"] is False
    assert flags["jitter"] == 0.2
