from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Windows UTF-8 fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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
    p.add_argument("--dhash-size", type=int, default=None, help="Kích thước dHash (mặc định lấy từ shared.yaml: 32)")
    p.add_argument("--dhash-hamming", type=int, default=None, help="Ngưỡng khoảng cách Hamming gom cụm (mặc định lấy từ shared.yaml: 370)")
    p.add_argument("--phash-hamming", type=int, default=None, help="Ngưỡng khoảng cách Hamming pHash gộp cụm (mặc định lấy từ shared.yaml: 10)")
    p.add_argument("--target-boxes", default=None, help="Ngưỡng box mục tiêu mỗi class trong train (mặc định lấy từ shared.yaml: 'max' để bằng class cao nhất)")
    args = p.parse_args(argv)

    cfg = resolve_config(
        pipeline_yaml=args.pipeline,
        shared_yaml=args.shared,
    )

    if args.dhash_size is not None:
        cfg.setdefault("split", {})["hash_size"] = args.dhash_size
    if args.dhash_hamming is not None:
        cfg.setdefault("split", {})["hamming_threshold"] = args.dhash_hamming
    if args.phash_hamming is not None:
        cfg.setdefault("split", {})["phash_threshold"] = args.phash_hamming
    if args.target_boxes is not None:
        cfg.setdefault("preprocess", {})["minority_target_boxes"] = args.target_boxes

    print(f"[run_preprocess] data_root     : {args.data_root}")
    print(f"[run_preprocess] image_size    : {cfg.get('preprocess', {}).get('image_size')}")
    print(f"[run_preprocess] min_box_px    : {cfg.get('preprocess', {}).get('min_box_px')}")
    print(f"[run_preprocess] dhash_size    : {cfg.get('split', {}).get('hash_size', 32)}")
    print(f"[run_preprocess] dhash_hamming : {cfg.get('split', {}).get('hamming_threshold', 370)}")
    print(f"[run_preprocess] phash_hamming : {cfg.get('split', {}).get('phash_threshold', 10)}")
    print(f"[run_preprocess] target_boxes  : {cfg.get('preprocess', {}).get('minority_target_boxes', 'max')}")

    out = run_offline(cfg, args.data_root)
    print(f"[run_preprocess] Processed cache sẵn sàng tại: {out}")


if __name__ == "__main__":
    main()
