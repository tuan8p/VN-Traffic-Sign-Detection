from __future__ import annotations
import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from vn_tsd.config.resolve import resolve_config
from vn_tsd.data.preprocess_offline import run_offline


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Offline preprocess (shared, detection boxes) → data/processed/"
    )
    p.add_argument("--data-root", required=True, help="Thư mục gốc dataset thô (chứa images/ và labels/)")
    p.add_argument("--shared", default="configs/shared.yaml", help="Đường dẫn shared.yaml")
    p.add_argument("--pipeline", default=None, help="Pipeline yaml để merge (tùy chọn)")
    args = p.parse_args(argv)

    cfg = resolve_config(
        pipeline_yaml=args.pipeline,
        shared_yaml=args.shared,
    )

    print(f"[run_preprocess] data_root  : {args.data_root}")
    print(f"[run_preprocess] image_size : {cfg.get('preprocess', {}).get('image_size')}")
    print(f"[run_preprocess] min_box_px : {cfg.get('preprocess', {}).get('min_box_px')}")

    out = run_offline(cfg, args.data_root)
    print(f"[run_preprocess] Processed cache sẵn sàng tại: {out}")


if __name__ == "__main__":
    main()
