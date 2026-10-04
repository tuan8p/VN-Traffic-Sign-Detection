from __future__ import annotations
from datetime import datetime
from pathlib import Path
import re
import uuid

def make_run_dir(outputs_root: str | Path, pipeline: str, run_name: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = run_name or uuid.uuid4().hex[:8]
    path = Path(outputs_root) / pipeline / f"{stamp}_{suffix}"
    for sub in ("logs", "figures", "checkpoints"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path

_RUN_HASH = re.compile(r"_[0-9a-f]{8}$")


def with_run_hash(name: str) -> str:
    """Append an 8-char hex so two W&B runs with the same config do not collide.

    _online, when used, stays before this hash. A name that already ends in
    _xxxxxxxx is left unchanged.
    """
    if _RUN_HASH.search(name):
        return name
    return f"{name}_{uuid.uuid4().hex[:8]}"
