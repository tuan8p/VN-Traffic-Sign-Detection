"""Shared offline preprocessing for detection.

# Windows UTF-8 fix
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

Keeps images + YOLO boxes (no classification crop). Writes a processed tree
both pipelines can read:

    data/processed/
      images/{train,val,test}/
      labels/{train,val,test}/
      data.yaml
      meta.json

Steps per image:
  1. Letterbox-resize to target (640×640 by default), pad với giá trị 114.
  2. Cập nhật tọa độ YOLO box theo tỉ lệ scale + offset padding.
  3. Loại bỏ box có min(w,h) < min_box_px (8px theo shared.yaml).
  4. Ghi file ảnh mới và file nhãn mới; sinh data.yaml và meta.json.

Run:
    python -m tools.run_preprocess --data-root data/raw
"""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_offline(cfg: dict[str, Any], data_root: str | Path) -> Path:
    """Chạy pipeline preprocessing offline.

    Args:
        cfg:       Resolved config dict (từ vn_tsd.config.resolve.resolve_config).
        data_root: Thư mục gốc dataset thô (chứa images/ và labels/).

    Returns:
        Path đến thư mục data/processed/ đã tạo.
    """
    data_root = Path(data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data_root không tồn tại: {data_root}")

    # ── Đọc config ──
    pre_cfg = cfg.get("preprocess", {})
    img_size_cfg = pre_cfg.get("image_size", [640, 640])
    if isinstance(img_size_cfg, int):
        target_h = target_w = img_size_cfg
    else:
        target_h, target_w = img_size_cfg[1], img_size_cfg[0]
    pad_value: int = int(pre_cfg.get("pad_value", 114))
    min_box_px: int = int(pre_cfg.get("min_box_px", 8))

    split_cfg = cfg.get("split", {})
    use_provided: bool = bool(split_cfg.get("use_provided_splits", True))
    ratios: dict[str, float] = split_cfg.get("ratios", {"train": 0.7, "val": 0.15, "test": 0.15})

    data_cfg = cfg.get("data", {})
    class_table_path = Path(data_cfg.get("class_table", "configs/classes.csv"))
    if not class_table_path.is_absolute():
        # Thử relative từ repo root
        _repo = Path(__file__).resolve().parents[2]
        class_table_path = _repo / class_table_path

    # Out dir
    _repo_root = Path(__file__).resolve().parents[2]
    out_root = _repo_root / "data" / "processed"
    out_root.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    rng = random.Random(seed)

    print(f"[preprocess] data_root  : {data_root}")
    print(f"[preprocess] out_root   : {out_root}")
    print(f"[preprocess] target size: {target_w}×{target_h}")
    print(f"[preprocess] min_box_px : {min_box_px}")
    print(f"[preprocess] pad_value  : {pad_value}")

    # ── Đọc classes ──
    classes = _load_classes(class_table_path)
    num_classes = len(classes)
    print(f"[preprocess] Số lớp     : {num_classes}")

    # ── Tìm ảnh + nhãn ──
    img_dir = data_root / "images"
    lbl_dir = data_root / "labels"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục images: {img_dir}")
    if not lbl_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục labels: {lbl_dir}")

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_paths = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in img_extensions
    ])
    print(f"[preprocess] Tổng ảnh gốc: {len(img_paths)}")

    # ── Phân chia split ──
    splits = _discover_splits(data_root, img_paths, use_provided, ratios, rng)
    for sname, stems in splits.items():
        print(f"[preprocess]   {sname}: {len(stems)} ảnh")

    # ── Tạo thư mục output ──
    for sname in splits:
        (out_root / "images" / sname).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / sname).mkdir(parents=True, exist_ok=True)

    # ── Xử lý từng split ──
    meta: dict[str, Any] = {
        "config": {
            "target_size": [target_w, target_h],
            "pad_value": pad_value,
            "min_box_px": min_box_px,
            "num_classes": num_classes,
            "seed": seed,
        },
        "splits": {},
    }

    t0 = time.time()
    stem_to_imgpath = {p.stem: p for p in img_paths}

    total_dropped_tiny = 0
    total_kept = 0

    for sname, stems in splits.items():
        n_imgs = 0
        n_boxes_in = 0
        n_boxes_out = 0
        n_tiny_dropped = 0
        n_no_label = 0

        for stem in stems:
            src_img = stem_to_imgpath.get(stem)
            if src_img is None:
                continue
            src_lbl = lbl_dir / (stem + ".txt")

            # Đọc nhãn gốc
            raw_boxes = _read_raw_labels(src_lbl)
            n_boxes_in += len(raw_boxes)

            # Đọc ảnh
            img = cv2.imread(str(src_img))
            if img is None:
                print(f"  [WARN] Không đọc được ảnh: {src_img}")
                continue

            orig_h, orig_w = img.shape[:2]

            # Letterbox
            lb_img, scale, dw, dh = _letterbox(img, target_w, target_h, pad_value)

            # Chuyển đổi box
            new_boxes = []
            for b in raw_boxes:
                cid, cx, cy, bw, bh = b
                nb = _transform_box(cx, cy, bw, bh, orig_w, orig_h, scale, dw, dh, target_w, target_h)
                if nb is None:
                    continue
                ncx, ncy, nw, nh = nb
                # Lọc tiny
                if nw * target_w < min_box_px or nh * target_h < min_box_px:
                    n_tiny_dropped += 1
                    continue
                new_boxes.append((cid, ncx, ncy, nw, nh))

            n_boxes_out += len(new_boxes)
            if len(raw_boxes) > 0 and len(new_boxes) == 0:
                n_no_label += 1

            # Ghi ảnh
            dst_img = out_root / "images" / sname / (stem + ".jpg")
            cv2.imwrite(str(dst_img), lb_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Ghi nhãn
            dst_lbl = out_root / "labels" / sname / (stem + ".txt")
            _write_labels(dst_lbl, new_boxes)

            n_imgs += 1

        meta["splits"][sname] = {
            "n_images": n_imgs,
            "n_boxes_in": n_boxes_in,
            "n_boxes_out": n_boxes_out,
            "n_tiny_dropped": n_tiny_dropped,
            "n_images_no_box_after_filter": n_no_label,
        }
        total_dropped_tiny += n_tiny_dropped
        total_kept += n_boxes_out
        print(
            f"[preprocess]   {sname}: {n_imgs} ảnh, "
            f"{n_boxes_out}/{n_boxes_in} box giữ lại "
            f"({n_tiny_dropped} tiny bỏ)"
        )

    elapsed = time.time() - t0
    meta["total_dropped_tiny"] = total_dropped_tiny
    meta["total_boxes_kept"] = total_kept
    meta["elapsed_seconds"] = round(elapsed, 2)

    # ── Ghi data.yaml ──
    _write_data_yaml(out_root, classes)
    print(f"[preprocess] data.yaml ghi tại: {out_root / 'data.yaml'}")

    # ── Ghi meta.json ──
    meta_path = out_root / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[preprocess] meta.json ghi tại: {meta_path}")
    print(f"[preprocess] Hoàn thành trong {elapsed:.1f}s  →  {out_root}")

    return out_root


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_classes(class_table: Path) -> dict[int, str]:
    """Đọc configs/classes.csv → {class_id: name_vi}."""
    classes: dict[int, str] = {}
    if not class_table.exists():
        raise FileNotFoundError(f"Không tìm thấy class table: {class_table}")
    with open(class_table, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            classes[int(row["class_id"])] = row["name_vi"]
    return classes


def _read_raw_labels(lbl_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Đọc file nhãn YOLO .txt. Trả về danh sách (class_id, cx, cy, w, h)."""
    if not lbl_path.exists():
        return []
    boxes = []
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cid = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append((cid, cx, cy, bw, bh))
            except ValueError:
                continue
    return boxes


def _letterbox(
    img: "np.ndarray",
    target_w: int,
    target_h: int,
    pad_value: int = 114,
) -> tuple["np.ndarray", float, float, float]:
    """Letterbox resize giữ nguyên tỉ lệ khung hình, bù padding màu pad_value.

    Returns:
        (lb_img, scale, dw, dh)  – scale là hệ số co dãn, dw/dh là offset padding (pixels).
    """
    orig_h, orig_w = img.shape[:2]
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = (target_w - new_w) / 2  # padding mỗi bên theo chiều ngang
    dh = (target_h - new_h) / 2  # padding mỗi bên theo chiều dọc

    top = round(dh - 0.1)
    bottom = round(dh + 0.1)
    left = round(dw - 0.1)
    right = round(dw + 0.1)

    lb_img = cv2.copyMakeBorder(
        img_resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )
    # Đảm bảo kích thước đúng (rounding edge case)
    lb_img = lb_img[:target_h, :target_w]
    return lb_img, scale, dw, dh


def _transform_box(
    cx: float, cy: float, bw: float, bh: float,
    orig_w: int, orig_h: int,
    scale: float, dw: float, dh: float,
    target_w: int, target_h: int,
) -> tuple[float, float, float, float] | None:
    """Chuyển đổi tọa độ box (normalized) sang tọa độ mới sau letterbox.

    Returns:
        (cx_new, cy_new, w_new, h_new) tất cả normalized [0,1], hoặc None nếu box invalid.
    """
    # Pixel coords trong ảnh gốc
    x1 = (cx - bw / 2) * orig_w
    y1 = (cy - bh / 2) * orig_h
    x2 = (cx + bw / 2) * orig_w
    y2 = (cy + bh / 2) * orig_h

    # Scale + offset
    x1 = x1 * scale + dw
    y1 = y1 * scale + dh
    x2 = x2 * scale + dw
    y2 = y2 * scale + dh

    # Clip vào [0, target]
    x1 = max(0.0, min(x1, target_w))
    y1 = max(0.0, min(y1, target_h))
    x2 = max(0.0, min(x2, target_w))
    y2 = max(0.0, min(y2, target_h))

    if x2 <= x1 or y2 <= y1:
        return None

    # Normalize
    ncx = (x1 + x2) / 2 / target_w
    ncy = (y1 + y2) / 2 / target_h
    nw  = (x2 - x1) / target_w
    nh  = (y2 - y1) / target_h

    return ncx, ncy, nw, nh


def _write_labels(dst: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    """Ghi file nhãn YOLO .txt."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for cid, cx, cy, bw, bh in boxes:
            f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def _write_data_yaml(out_root: Path, classes: dict[int, str]) -> None:
    """Ghi data.yaml theo chuẩn Ultralytics YOLO."""
    import yaml  # pyYAML
    names = {i: classes[i] for i in sorted(classes.keys())}
    data = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": names,
    }
    with open(out_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _find_split_file(split_dir: Path, sname: str) -> Path | None:
    """Tìm file split phù hợp trong thư mục split_dataset/."""
    candidates = [
        split_dir / f"{sname}.txt",
        split_dir / f"{sname}_files.txt",
        split_dir / f"{sname}ing.txt",
        split_dir / f"{sname}ing_files.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    for f in split_dir.glob("*.txt"):
        if sname.lower() in f.stem.lower():
            return f
    return None


def _discover_splits(
    data_root: Path,
    img_paths: list[Path],
    use_provided: bool,
    ratios: dict[str, float],
    rng: random.Random,
) -> dict[str, list[str]]:
    """Tìm / tạo phân chia train/val/test. Trả về {split: [stem, ...]}."""
    stems_all = [p.stem for p in img_paths]

    # 1. Dùng split có sẵn trong split_dataset/
    if use_provided:
        split_dir = data_root / "split_dataset"
        splits_found: dict[str, list[str]] = {}
        for sname in ("train", "val", "test"):
            sf = _find_split_file(split_dir, sname)
            if not sf:
                continue
            with open(sf, "r", encoding="utf-8") as fp:
                lines = [l.strip() for l in fp if l.strip()]
            stems_in_split = [Path(line).stem for line in lines]
            valid = [s for s in stems_in_split if s in {p.stem for p in img_paths}]
            if valid:
                splits_found[sname] = valid

        if "train" in splits_found:
            print(f"[preprocess] Dùng split có sẵn từ {split_dir}")
            # Nếu thiếu val: trích xuất 15% từ train làm val để đủ 3 tập
            if "val" not in splits_found:
                tr = splits_found["train"]
                n_val = max(1, round(len(tr) * ratios.get("val", 0.15)))
                shuffled_tr = tr[:]
                rng.shuffle(shuffled_tr)
                splits_found["val"] = shuffled_tr[:n_val]
                splits_found["train"] = shuffled_tr[n_val:]
                print(f"[preprocess]   Tự động trích {n_val} ảnh từ train làm tập val để phục vụ validation.")

            # Nếu thiếu test: trích xuất hoặc dùng val
            if "test" not in splits_found:
                splits_found["test"] = splits_found["val"]

            return splits_found
        print(f"[preprocess] Không tìm thấy split có sẵn trong {split_dir}, tự chia theo tỉ lệ.")

    # 2. Tự chia ngẫu nhiên
    shuffled = stems_all[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * ratios.get("train", 0.7))
    n_val = round(n * ratios.get("val", 0.15))

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

