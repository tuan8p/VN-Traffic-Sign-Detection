"""Shared offline preprocessing for detection.

# Windows UTF-8 fix
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

Tạo bộ dữ liệu chuẩn hoá cho cả 2 pipeline tại:
    data/processed/
      images/{train,val,test}/
      labels/{train,val,test}/
      data.yaml
      meta.json
      {train,val,test}_files.txt

Quy trình chuẩn hóa trực tiếp từ dữ liệu ảnh (Dynamic Image Clustering & Split):
  1. Chỉ dựa vào toàn bộ ảnh thô trong images/: Tuyệt đối không đọc hay dựa vào bất kỳ file split có sẵn nào trong split_dataset/.
  2. BƯỚC 1: Phân cụm (Clustering):
     - Tính Difference Hash 1024-bit (size=32, Hamming <= 370) cho toàn bộ ảnh thô.
     - Gom toàn bộ ảnh thành các cụm tương đồng bối cảnh/chuỗi thời gian bằng thuật toán Disjoint Set Union (DSU).
  3. BƯỚC 2: Phân chia tập (Dataset Splitting theo tỉ lệ 70% Train / 15% Val / 15% Test):
     - Dựa vào kết quả phân cụm ở Bước 1: gán NGUYÊN VẸN từng cụm vào một tập duy nhất.
     - ĐẢM BẢO TUYỆT ĐỐI: Không được lấy các ảnh của cùng 1 cụm mà nằm trong 2 tập khác nhau (100% Cluster Isolation).
     - Đảm bảo tập Train có đủ toàn bộ 52/52 lớp nhãn (ưu tiên cụm chứa lớp độc nhất vào Train).
     - Phân bổ đều ảnh tối (brightness < 85) và ảnh mờ (sharpness < 100) để cân bằng quang học giữa các tập.
  4. BƯỚC 3: Chuẩn hoá kích thước (Letterbox Resize 640×640):
     - Giữ nguyên tỉ lệ khung hình (aspect ratio), bù padding xám 114.
     - Chuyển đổi toạ độ YOLO box và loại bỏ triệt để các box siêu nhỏ (< min_box_px, mặc định 8px).
  5. BƯỚC 4: Cân bằng nhãn thiểu số cho tập Train (Data Augmentation):
     - Tăng cường dữ liệu cho các nhãn dưới 50 box trong tập Train.
     - Kiểm tra đối soát dHash không gây rò rỉ sang tập Val và Test.
  6. BƯỚC 5: Xuất metadata và kiểm tra rò rỉ dữ liệu (Data Leakage Verification):
     - Ghi nhận báo cáo vào data/processed/meta.json, data.yaml, và lưu danh sách file split tại data/processed/.
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image

from analysis.image_utils import calculate_dhash
from analysis.leakage import find_leaked_test_stems
from analysis.split import split_by_dhash_clusters
from vn_tsd.data.augmentation import augment_minority_classes, _read_raw_labels, _write_labels


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



def _letterbox_image(
    img: np.ndarray,
    target_size: tuple[int, int] = (640, 640),
    pad_value: int = 114,
) -> tuple[np.ndarray, float, float, float]:
    """Resize giữ nguyên tỉ lệ khung hình và padding viền màu xám 114."""
    th, tw = target_size
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), pad_value, dtype=np.uint8)

    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    canvas[dy : dy + nh, dx : dx + nw] = resized
    return canvas, scale, float(dx), float(dy)


def _adjust_boxes(
    boxes: list[tuple[int, float, float, float, float]],
    orig_w: int,
    orig_h: int,
    scale: float,
    dx: float,
    dy: float,
    target_w: int,
    target_h: int,
    min_box_px: int = 8,
) -> tuple[list[tuple[int, float, float, float, float]], int]:
    """Chuyển đổi toạ độ YOLO box sang canvas mới và loại bỏ các box < min_box_px."""
    adjusted = []
    dropped_tiny = 0

    for cid, cx, cy, bw, bh in boxes:
        abs_cx = cx * orig_w
        abs_cy = cy * orig_h
        abs_bw = bw * orig_w
        abs_bh = bh * orig_h

        new_cx = (abs_cx * scale + dx) / target_w
        new_cy = (abs_cy * scale + dy) / target_h
        new_bw = (abs_bw * scale) / target_w
        new_bh = (abs_bh * scale) / target_h

        px_w = new_bw * target_w
        px_h = new_bh * target_h
        if px_w < min_box_px or px_h < min_box_px:
            dropped_tiny += 1
            continue

        x1 = max(0.0, new_cx - new_bw / 2)
        y1 = max(0.0, new_cy - new_bh / 2)
        x2 = min(1.0, new_cx + new_bw / 2)
        y2 = min(1.0, new_cy + new_bh / 2)
        if x2 <= x1 or y2 <= y1:
            continue

        final_cx = (x1 + x2) / 2
        final_cy = (y1 + y2) / 2
        final_bw = x2 - x1
        final_bh = y2 - y1

        adjusted.append((cid, final_cx, final_cy, final_bw, final_bh))

    return adjusted, dropped_tiny


def _write_data_yaml(out_root: Path, classes: dict[int, str]) -> None:
    """Sinh data.yaml chuẩn Ultralytics YOLO."""
    names_dict = {cid: classes[cid] for cid in sorted(classes.keys())}
    content = [
        f"path: {out_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(classes)}",
        f"names: {json.dumps(names_dict, ensure_ascii=False)}",
        "",
    ]
    with open(out_root / "data.yaml", "w", encoding="utf-8") as f:
        f.write("\n".join(content))


def preprocess_dataset_offline(
    data_root: str | Path,
    out_root: str | Path = "data/processed",
    class_table: str | Path = "configs/classes.csv",
    image_size: tuple[int, int] = (640, 640),
    min_box_px: int = 8,
    pad_value: int = 114,
    augment_minority: bool = True,
    minority_target_boxes: int = 50,
    ratio: tuple[float, float, float] = (0.70, 0.15, 0.15),
    hash_size: int = 32,
    hamming_thresh: int = 370,
    seed: int = 42,
) -> Path:
    """Pipeline tiền xử lý offline: phân cụm toàn bộ ảnh thô và chia split theo cụm độc lập."""
    data_root = Path(data_root)
    out_root = Path(out_root)
    class_table = Path(class_table)

    print(f"[preprocess] data_root  : {data_root.resolve()}")
    print(f"[preprocess] out_root   : {out_root.resolve()}")
    print(f"[preprocess] target size: {image_size[0]}×{image_size[1]}")
    print(f"[preprocess] min_box_px : {min_box_px}")
    print(f"[preprocess] pad_value  : {pad_value}")

    classes = _load_classes(class_table)
    print(f"[preprocess] Số lớp     : {len(classes)}")

    img_dir = data_root / "images"
    lbl_dir = data_root / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy images/ hoặc labels/ tại {data_root}")

    all_img_paths = sorted(img_dir.glob("*.jpg"))
    if not all_img_paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh *.jpg nào trong {img_dir}")
    print(f"[preprocess] Tổng ảnh gốc: {len(all_img_paths):,}")

    stem_to_imgpath = {p.stem: p for p in all_img_paths}
    all_stems = sorted(stem_to_imgpath.keys())

    # ── BƯỚC 1 & 2: Phân cụm dHash trực tiếp từ ảnh và chia split theo cụm (không dùng split có sẵn) ──
    print("[preprocess] ========================================================")
    print(f"[preprocess] BƯỚC 1: Phân cụm toàn bộ ảnh thô bằng dHash (size={hash_size}, Hamming <= {hamming_thresh})...")
    print(f"[preprocess] BƯỚC 2: Phân chia các cụm vào Train, Val, Test theo tỉ lệ ({ratio[0]*100:.0f}/{ratio[1]*100:.0f}/{ratio[2]*100:.0f})...")
    print("[preprocess]         QUY TẮC: Mọi ảnh trong cùng 1 cụm PHẢI thuộc cùng 1 tập (không chia rẽ cụm)!")
    print("[preprocess] ========================================================")

    splits = split_by_dhash_clusters(
        img_paths=all_img_paths,
        lbl_dir=lbl_dir,
        ratio=ratio,
        hash_size=hash_size,
        hamming_thresh=hamming_thresh,
        seed=seed,
    )

    # Kiểm tra an toàn: Đảm bảo các tập hoàn toàn không giao nhau (rời rạc tuyệt đối)
    tr_set = set(splits["train"])
    va_set = set(splits["val"])
    te_set = set(splits["test"])
    assert tr_set.isdisjoint(va_set), "LỖI: Có ảnh xuất hiện đồng thời trong Train và Val!"
    assert tr_set.isdisjoint(te_set), "LỖI: Có ảnh xuất hiện đồng thời trong Train và Test!"
    assert va_set.isdisjoint(te_set), "LỖI: Có ảnh xuất hiện đồng thời trong Val và Test!"
    assert len(tr_set) + len(va_set) + len(te_set) == len(all_stems), "LỖI: Tổng số ảnh split không khớp tổng số ảnh gốc!"
    print(f"[preprocess] Xác nhận phân tách hoàn tất: Train={len(tr_set):,}, Val={len(va_set):,}, Test={len(te_set):,} (100% độc lập cụm).")

    # Lưu lại danh sách split vừa phân chia vào out_root (data/processed/), tuyệt đối không ghi vào data/raw
    split_save_dir = out_root / "split_files"
    split_save_dir.mkdir(parents=True, exist_ok=True)
    for sname in ("train", "val", "test"):
        lines = [f"{s}.jpg\n" for s in splits[sname]]
        with open(split_save_dir / f"{sname}.txt", "w", encoding="utf-8") as f:
            f.writelines(lines)
        with open(out_root / f"{sname}_files.txt", "w", encoding="utf-8") as f:
            f.writelines(lines)

    # ── Xoá sạch thư mục đích để tránh file cũ còn sót lại ──
    for sub in ("images", "labels"):
        d = out_root / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        for sname in ("train", "val", "test"):
            (d / sname).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    meta: dict[str, Any] = {
        "dataset_name": "VN-Traffic-Sign-Detection-Processed",
        "source_data_root": str(data_root),
        "target_image_size": list(image_size),
        "min_box_px": min_box_px,
        "pad_value": pad_value,
        "num_classes": len(classes),
        "splits": {},
        "total_images_in": len(all_stems),
    }

    total_kept = 0
    total_dropped_tiny = 0

    # ── Xử lý letterbox và lọc box nhỏ trên từng split ──
    for sname, stems in splits.items():
        dst_img_dir = out_root / "images" / sname
        dst_lbl_dir = out_root / "labels" / sname

        n_boxes_in = 0
        n_boxes_out = 0
        n_tiny_dropped = 0

        for stem in stems:
            src_img_p = stem_to_imgpath[stem]
            src_lbl_p = lbl_dir / f"{stem}.txt"

            raw_boxes = _read_raw_labels(src_lbl_p)
            n_boxes_in += len(raw_boxes)

            img = cv2.imread(str(src_img_p))
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]
            canvas, scale, dx, dy = _letterbox_image(img, image_size, pad_value)

            adj_boxes, tiny_cnt = _adjust_boxes(
                raw_boxes, orig_w, orig_h, scale, dx, dy, image_size[0], image_size[1], min_box_px
            )
            n_tiny_dropped += tiny_cnt
            n_boxes_out += len(adj_boxes)

            cv2.imwrite(str(dst_img_dir / f"{stem}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            _write_labels(dst_lbl_dir / f"{stem}.txt", adj_boxes)

        meta["splits"][sname] = {
            "n_images": len(stems),
            "n_boxes_in": n_boxes_in,
            "n_boxes_out": n_boxes_out,
            "n_tiny_dropped": n_tiny_dropped,
        }
        total_kept += n_boxes_out
        total_dropped_tiny += n_tiny_dropped
        print(f"[preprocess]   {sname}: {len(stems)} ảnh, {n_boxes_out}/{n_boxes_in} box giữ lại ({n_tiny_dropped} tiny bỏ)")

    # ── Tăng cường dữ liệu cho các lớp thiểu số trong tập train (nếu bật) ──
    if augment_minority and "train" in splits:
        # Tiền tính toán mã dHash của val và test để kiểm tra rò rỉ trong quá trình sinh ảnh augmented
        val_test_hashes = []
        for s in ("val", "test"):
            s_dir = out_root / "images" / s
            for p in s_dir.glob("*.jpg"):
                h = calculate_dhash(p, hash_size=32, strip_padding=True)
                if h is not None:
                    val_test_hashes.append(h)

        val_test_packed = np.packbits(np.array(val_test_hashes, dtype=np.uint8), axis=1) if val_test_hashes else None

        aug_stats = augment_minority_classes(
            train_img_dir=out_root / "images" / "train",
            train_lbl_dir=out_root / "labels" / "train",
            val_test_packed=val_test_packed,
            target_min_boxes=minority_target_boxes,
            min_box_px=min_box_px,
            seed=seed,
        )
        meta["splits"]["train"]["n_images"] += aug_stats["n_augmented_images"]
        meta["splits"]["train"]["n_boxes_out"] += aug_stats["n_augmented_boxes"]
        meta["minority_augmentation"] = aug_stats
        total_kept += aug_stats["n_augmented_boxes"]

    elapsed = time.time() - t0
    meta["total_dropped_tiny"] = total_dropped_tiny
    meta["total_boxes_kept"] = total_kept
    meta["elapsed_seconds"] = round(elapsed, 2)

    # ── Khảo sát rò rỉ dữ liệu Test vs Train và ghi test_clean_files.txt ──
    train_stems_processed = [p.stem for p in (out_root / "images" / "train").glob("*.jpg")]
    test_stems_processed = [p.stem for p in (out_root / "images" / "test").glob("*.jpg")]
    proc_stem_to_path = {p.stem: p for p in out_root.glob("images/*/*.jpg")}

    clean_test, leaked_test = find_leaked_test_stems(
        train_stems=train_stems_processed,
        test_stems=test_stems_processed,
        stem_to_path=proc_stem_to_path,
        hash_size=32,
        hamming_thresh=370,
    )

    clean_file = out_root / "test_clean_files.txt"
    leak_file = out_root / "test_leaked_files.txt"
    with open(clean_file, "w", encoding="utf-8") as f:
        f.write("\n".join(clean_test) + "\n")
    with open(leak_file, "w", encoding="utf-8") as f:
        f.write("\n".join(leaked_test) + "\n")

    meta["leakage_analysis"] = {
        "total_test": len(test_stems_processed),
        "clean_test_count": len(clean_test),
        "leaked_test_count": len(leaked_test),
        "leaked_pct": round(len(leaked_test) / max(len(test_stems_processed), 1) * 100, 2),
        "status": "PASSED" if not leaked_test else "LEAK_DETECTED",
    }
    print(f"[preprocess] Kiểm tra Data Leakage (dHash size=32, Hamming ≤ 370): {len(leaked_test)}/{len(test_stems_processed)} ảnh rò rỉ.")
    print(f"[preprocess]   Ghi danh sách test sạch : {clean_file}")
    print(f"[preprocess]   Ghi danh sách rò rỉ     : {leak_file}")

    # ── Ghi data.yaml và meta.json ──
    _write_data_yaml(out_root, classes)
    print(f"[preprocess] data.yaml ghi tại: {out_root / 'data.yaml'}")

    meta_path = out_root / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[preprocess] meta.json ghi tại: {meta_path}")
    print(f"[preprocess] Hoàn thành trong {elapsed:.1f}s  →  {out_root}")

    return out_root


def run_offline(cfg: dict[str, Any], data_root: str | Path | None = None) -> Path:
    """Wrapper chính nhận config dictionary từ tools/run_preprocess.py."""
    prep = cfg.get("preprocess", {})
    split_cfg = cfg.get("split", {})
    ratios_dict = split_cfg.get("ratios", {"train": 0.70, "val": 0.15, "test": 0.15})
    ratios = (
        float(ratios_dict.get("train", 0.70)),
        float(ratios_dict.get("val", 0.15)),
        float(ratios_dict.get("test", 0.15)),
    )
    hash_sz = int(split_cfg.get("hash_size", 32))
    h_thresh = int(split_cfg.get("hamming_threshold", 370))

    d_root = data_root or prep.get("raw_dir", "data/raw")
    out_root = prep.get("processed_dir", "data/processed")
    target_size = tuple(prep.get("image_size", [640, 640]))
    min_box_px = int(prep.get("min_box_px", 8))
    pad_val = int(prep.get("pad_value", 114))
    aug_min = bool(cfg.get("augment_minority", True))
    aug_target = int(cfg.get("minority_target_boxes", 50))
    seed = int(cfg.get("seed", 42))

    return preprocess_dataset_offline(
        data_root=d_root,
        out_root=out_root,
        image_size=target_size,
        min_box_px=min_box_px,
        pad_value=pad_val,
        augment_minority=aug_min,
        minority_target_boxes=aug_target,
        ratio=ratios,
        hash_size=hash_sz,
        hamming_thresh=h_thresh,
        seed=seed,
    )
