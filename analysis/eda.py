"""Module điều phối trung tâm phân tích khám phá dữ liệu (EDA Router).

Tự động nhận diện định dạng dữ liệu (RAW hay PROCESSED) và chuyển tiếp tới module chuyên trách:
  - analysis.eda_raw: Chuyên phân tích tập thô (8 biểu đồ chuẩn, Figure 01 - 08)
  - analysis.eda_processed: Chuyên phân tích tập sau tiền xử lý (9 biểu đồ chuẩn, Figure 01 - 09)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from analysis.eda_raw import analyze_raw_dataset
from analysis.eda_processed import analyze_processed_dataset

__all__ = [
    "analyze_raw_dataset",
    "analyze_processed_dataset",
    "main",
]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Offline EDA toàn diện chuẩn 8 biểu đồ (RAW) hoặc 9 biểu đồ (PROCESSED) cho VN-Traffic-Sign-Detection"
    )
    p.add_argument("--data-root", default=None, help="Thư mục gốc dataset (raw hoặc processed)")
    p.add_argument("--figures-dir", default=None, help="Thư mục xuất hình (mặc định: analysis/figures/)")
    p.add_argument("--class-table", default=None, help="Đường dẫn configs/classes.csv")
    p.add_argument("--image-size", type=int, default=640, help="Kích thước ảnh để tính px (default: 640)")
    p.add_argument("--min-box-px", type=int, default=8, help="Ngưỡng tiny box (default: 8)")
    p.add_argument("--num-samples", type=int, default=9, help="Số ảnh mẫu vẽ demo nhãn (default: 9)")
    p.add_argument("--quality-samples", type=int, default=1000, help="Số ảnh lấy mẫu đo độ sáng/độ nét (default: 1000)")
    p.add_argument("--dhash-size", type=int, default=32, help="Kích thước cạnh dHash (default: 32 -> 1024 bits)")
    p.add_argument("--dhash-hamming", type=int, default=380, help="Ngưỡng khoảng cách Hamming dHash gom cụm (default: 380)")
    p.add_argument("--phash-hamming", type=int, default=18, help="Ngưỡng khoảng cách Hamming pHash gộp cụm (default: 18)")
    args = p.parse_args(argv)

    data_root_str = args.data_root or os.environ.get("DATA_ROOT") or ""
    if not data_root_str:
        print("[EDA] Chưa có --data-root. Dùng: python -m analysis.eda --data-root data/raw")
        sys.exit(1)
    data_root = Path(data_root_str).resolve()
    if not data_root.exists():
        print(f"[EDA] Không tìm thấy thư mục: {data_root}")
        sys.exit(1)

    is_processed = (data_root / "labels" / "train").is_dir() or "processed" in data_root.parts

    if is_processed:
        analyze_processed_dataset(
            data_root=data_root,
            figures_dir=args.figures_dir,
            class_table=args.class_table,
            image_size=args.image_size,
            min_box_px=args.min_box_px,
            num_samples=args.num_samples,
            quality_samples=args.quality_samples,
            dhash_size=args.dhash_size,
            dhash_hamming=args.dhash_hamming,
            phash_hamming=args.phash_hamming,
        )
    else:
        analyze_raw_dataset(
            data_root=data_root,
            figures_dir=args.figures_dir,
            class_table=args.class_table,
            image_size=args.image_size,
            min_box_px=args.min_box_px,
            num_samples=args.num_samples,
            quality_samples=args.quality_samples,
            dhash_size=args.dhash_size,
            dhash_hamming=args.dhash_hamming,
            phash_hamming=args.phash_hamming,
        )


if __name__ == "__main__":
    main()
