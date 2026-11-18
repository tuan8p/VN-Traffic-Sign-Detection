"""Optional online preprocessing toggles for detection pipelines.

Offline preprocessing (resize policy, class map, split files) is shared and
cached. Online ops run per-batch while tuning a pipeline and default OFF.
Each pipeline yaml may override `online_preprocess`. The master switch
`enabled` gates every op: if it is false, individual flags are ignored.
"""
from __future__ import annotations
from typing import Any

TOGGLE_KEYS = (
    "letterbox",
    "normalize",
    "hflip",
    "color_jitter",
    "mosaic",
    "mixup",
)

def resolve_online_preprocess(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = dict(cfg.get("online_preprocess") or {})
    enabled = bool(raw.get("enabled", False))
    out: dict[str, Any] = {"enabled": enabled}
    for key in TOGGLE_KEYS:
        out[key] = bool(raw.get(key, False)) if enabled else False
    if enabled and "jitter" in raw:
        out["jitter"] = float(raw.get("jitter", 0.0))
    else:
        out["jitter"] = 0.0
    return out

def describe_online_preprocess(cfg: dict[str, Any]) -> str:
    flags = resolve_online_preprocess(cfg)
    if not flags["enabled"]:
        return "online_preprocess: OFF"
    on = [k for k in TOGGLE_KEYS if flags[k]]
    return "online_preprocess: ON [" + (", ".join(on) if on else "no ops") + "]"
