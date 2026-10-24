"""Tiền xử lý offline 5 bước chuẩn hoá dataset cho Faster R-CNN và YOLO26m."""
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

from collections import defaultdict, Counter

from analysis.image_utils import calculate_dhash
from analysis.clustering import cluster_images_dhash
from vn_tsd.data.split import split_clusters_for_classes, split_by_dhash_clusters
from vn_tsd.data.augmentation import (
    augment_minority_classes,
    _read_raw_labels,
    _write_labels,
    get_safe_crop_bounds,
    transform_and_letterbox_crop,
)


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
) -> list[tuple[int, float, float, float, float]]:
    """Chuyển đổi toạ độ YOLO box sang canvas mới sau letterbox (không lọc tiny vì đã lọc sạch ở Bước 1)."""
    adjusted = []

    for cid, cx, cy, bw, bh in boxes:
        abs_cx = cx * orig_w
        abs_cy = cy * orig_h
        abs_bw = bw * orig_w
        abs_bh = bh * orig_h

        new_cx = (abs_cx * scale + dx) / target_w
        new_cy = (abs_cy * scale + dy) / target_h
        new_bw = (abs_bw * scale) / target_w
        new_bh = (abs_bh * scale) / target_h

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

    return adjusted


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

def _filter_tiny_boxes(
    img_paths: list[Path],
    lbl_dir: Path,
    min_box_px: int = 8,
    target_size: tuple[int, int] = (640, 640),
) -> tuple[dict[str, list[tuple[int, float, float, float, float]]], int, int]:
    """BƯỚC 1: Lọc bỏ toàn bộ box nhỏ (< min_box_px) trên canvas target_size."""
    clean_boxes = {}
    total_in = 0
    total_dropped = 0
    tw, th = target_size

    for p in img_paths:
        stem = p.stem
        raw_b = _read_raw_labels(lbl_dir / f"{stem}.txt")
        total_in += len(raw_b)

        orig_w, orig_h = 960, 540
        try:
            with Image.open(p) as im:
                orig_w, orig_h = im.size
        except Exception:
            pass
        scale = min(tw / orig_w, th / orig_h)

        kept = []
        for cid, cx, cy, bw, bh in raw_b:
            if (bw * orig_w * scale >= min_box_px) and (bh * orig_h * scale >= min_box_px):
                kept.append((cid, cx, cy, bw, bh))
            else:
                total_dropped += 1
        clean_boxes[stem] = kept

    return clean_boxes, total_in, total_dropped


def preprocess_dataset_offline(
    data_root: str | Path = "data/raw",
    out_root: str | Path = "data/processed",
    class_table: str | Path = "configs/classes.csv",
    image_size: tuple[int, int] = (640, 640),
    min_box_px: int = 8,
    pad_value: int = 114,
    augment_minority: bool = True,
    minority_target_boxes: int | str = "adaptive",
    ratio: tuple[float, float, float] = (0.70, 0.15, 0.15),
    hash_size: int = 32,
    hamming_thresh: int = 370,
    phash_thresh: int | None = 10,
    seed: int = 42,
) -> Path:
    """Pipeline tiền xử lý offline 5 BƯỚC chuẩn xác:
      B1: Xóa các box nhỏ (< min_box_px)
      B2: Tính gom cụm (dHash + pHash) và chia
      B3: Đảm bảo train luôn có đầy đủ class (52/52 lớp, 100% Cluster Isolation)
      B4: Tạo augment trong train (cân bằng nhãn thiểu số)
      B5: Resize ảnh (Letterbox 640x640) & xuất data.yaml, meta.json
    """
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
    t0 = time.time()

    # ── BƯỚC 1: Xóa các bounding box nhỏ (< min_box_px) ──
    print("[preprocess] ========================================================")
    print(f"[preprocess] BƯỚC 1: Xoá các bounding box nhỏ (< {min_box_px}px)...")
    clean_boxes, total_boxes_in, total_dropped_tiny = _filter_tiny_boxes(
        all_img_paths, lbl_dir, min_box_px=min_box_px, target_size=image_size
    )
    total_valid_boxes = sum(len(b) for b in clean_boxes.values())
    print(f"[preprocess]   Đã quét {len(all_img_paths):,} ảnh: loại bỏ {total_dropped_tiny:,} box nhỏ (< {min_box_px}px), giữ lại {total_valid_boxes:,} box hợp lệ.")

    # ── BƯỚC 2: Tính gom cụm ──
    print("[preprocess] ========================================================")
    ph_str = f" + pHash 64-bit (Hamming <= {phash_thresh})" if phash_thresh else ""
    print(f"[preprocess] BƯỚC 2: Tính gom cụm bằng dHash 1024-bit (size={hash_size}, Hamming <= {hamming_thresh}){ph_str}...")
    cluster_res = cluster_images_dhash(
        all_img_paths,
        hash_size=hash_size,
        hamming_thresh=hamming_thresh,
        phash_thresh=phash_thresh,
    )
    clusters = cluster_res.get("clusters", {})
    print(f"[preprocess]   Tìm thấy {len(clusters):,} cụm tương đồng bối cảnh từ {len(all_img_paths):,} ảnh thô.")

    # ── BƯỚC 3: Đảm bảo train luôn có đầy đủ class ──
    print("[preprocess] ========================================================")
    print(f"[preprocess] BƯỚC 3: Đảm bảo tập Train luôn có đầy đủ class (52/52 lớp, 100% Cluster Isolation)...")
    stem_to_classes = {s: [b[0] for b in boxes] for s, boxes in clean_boxes.items()}
    splits = split_clusters_for_classes(
        clusters=clusters,
        stem_to_classes=stem_to_classes,
        img_paths=all_img_paths,
        ratio=ratio,
        seed=seed,
    )

    tr_set = set(splits["train"])
    va_set = set(splits["val"])
    te_set = set(splits["test"])
    assert tr_set.isdisjoint(va_set), "LỖI: Có ảnh xuất hiện đồng thời trong Train và Val!"
    assert tr_set.isdisjoint(te_set), "LỖI: Có ảnh xuất hiện đồng thời trong Train và Test!"
    assert va_set.isdisjoint(te_set), "LỖI: Có ảnh xuất hiện đồng thời trong Val và Test!"
    assert len(tr_set) + len(va_set) + len(te_set) == len(all_stems), "LỖI: Tổng số ảnh split không khớp tổng số ảnh gốc!"

    tr_classes = {b[0] for s in splits["train"] for b in clean_boxes[s]}
    assert len(tr_classes) == len(classes), f"LỖI: Tập train chỉ có {len(tr_classes)}/{len(classes)} lớp!"
    print(f"[preprocess]   Phân bổ hoàn tất: Train={len(tr_set):,} ảnh (đủ {len(tr_classes)}/52 lớp), Val={len(va_set):,} ảnh, Test={len(te_set):,} ảnh (100% độc lập cụm).")

    # Lưu lại danh sách split vừa phân chia vào out_root (data/processed/)
    split_save_dir = out_root / "split_files"
    split_save_dir.mkdir(parents=True, exist_ok=True)
    for sname in ("train", "val", "test"):
        lines = [f"{s}.jpg\n" for s in splits[sname]]
        with open(split_save_dir / f"{sname}.txt", "w", encoding="utf-8") as f:
            f.writelines(lines)
        with open(out_root / f"{sname}_files.txt", "w", encoding="utf-8") as f:
            f.writelines(lines)

    # ── BƯỚC 4: Tạo augment trong tập Train (cân bằng nhãn thiểu số) ──
    print("[preprocess] ========================================================")
    print(f"[preprocess] BƯỚC 4: Tạo augment trong tập Train (cân bằng nhãn thiểu số)...")
    aug_items: list[tuple[str, np.ndarray, list[tuple[int, float, float, float, float]]]] = []
    aug_stats = {"n_augmented_images": 0, "n_augmented_boxes": 0}

    if augment_minority and "train" in splits:
        rng = random.Random(seed)
        train_class_counts: Counter[int] = Counter()
        class_candidates: dict[int, list[tuple[str, tuple[int, float, float, float, float], tuple[float, float, float, float]]]] = defaultdict(list)

        for s in splits["train"]:
            boxes = clean_boxes[s]
            p = stem_to_imgpath[s]
            orig_w, orig_h = 960, 540
            try:
                with Image.open(p) as im:
                    orig_w, orig_h = im.size
            except Exception:
                pass

            for i, b in enumerate(boxes):
                train_class_counts[b[0]] += 1
                other_b = [other for j, other in enumerate(boxes) if j != i and other[0] != b[0]]
                bounds = get_safe_crop_bounds(b, other_b, orig_w, orig_h)
                if bounds is not None:
                    class_candidates[b[0]].append((s, b, bounds))

        # Xác định mục tiêu augment cho từng class
        class_targets: dict[int, int] = {}
        policy_str = str(minority_target_boxes).strip().lower()
        if policy_str in ("adaptive", "ratio", "tier", "multiplier"):
            for cid, cnt in train_class_counts.items():
                if cnt < 30:
                    class_targets[cid] = max(int(round(cnt * 2.5)), min(15, cnt * 3))
                elif cnt <= 150:
                    class_targets[cid] = int(round(cnt * 1.75))
                else:
                    class_targets[cid] = cnt
        elif policy_str in ("max", "-1", "none"):
            max_c = max(train_class_counts.values()) if train_class_counts else 0
            for cid in train_class_counts:
                class_targets[cid] = max_c
        else:
            fixed_t = int(minority_target_boxes)
            for cid, cnt in train_class_counts.items():
                class_targets[cid] = max(cnt, fixed_t)

        max_augmented_per_class = 7
        class_needed = {
            c: min(class_targets[c] - train_class_counts[c], max_augmented_per_class)
            for c in train_class_counts
            if class_targets[c] > train_class_counts[c]
        }

        print(f"[preprocess]   Cần sinh thêm cho {len(class_needed)} lớp thiểu số trong Train (trần tối đa: {max_augmented_per_class} ảnh/lớp)...")

        # Tiền tính dHash của val và test để chống rò rỉ dữ liệu
        val_test_hashes = []
        for sname in ("val", "test"):
            for s in splits[sname]:
                h = calculate_dhash(stem_to_imgpath[s], hash_size=32, strip_padding=True)
                if h is not None:
                    val_test_hashes.append(h)
        val_test_packed = np.packbits(np.array(val_test_hashes, dtype=np.uint8), axis=1) if val_test_hashes else None
        popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

        img_cache: dict[str, np.ndarray] = {}
        def _get_img(st: str) -> np.ndarray | None:
            if st in img_cache:
                return img_cache[st]
            im = cv2.imread(str(stem_to_imgpath[st]))
            if len(img_cache) < 200:
                img_cache[st] = im
            return im

        aug_per_class: Counter[int] = Counter()
        for cid, needed in class_needed.items():
            candidates = class_candidates.get(cid, [])
            if not candidates:
                continue
            cands_pool = list(candidates)
            rng.shuffle(cands_pool)
            gen = 0
            attempts = 0
            while gen < needed and attempts < needed * 20:
                attempts += 1
                stem, b, bounds = cands_pool[(attempts - 1) % len(cands_pool)]
                img = _get_img(stem)
                if img is None:
                    continue
                canvas, final_box = transform_and_letterbox_crop(
                    img=img,
                    target_box=b,
                    safe_bounds=bounds,
                    rng=rng,
                    target_size=image_size,
                    min_box_px=min_box_px,
                    pad_value=pad_value,
                )
                if canvas is None or final_box is None:
                    continue

                if val_test_packed is not None:
                    h_aug = calculate_dhash(canvas, hash_size=32, strip_padding=True)
                    if h_aug is not None:
                        packed_aug = np.packbits(h_aug)
                        diff = np.bitwise_xor(val_test_packed, packed_aug)
                        dists = popcount[diff].sum(axis=1)
                        if dists.min() <= 374:
                            continue

                gen += 1
                aug_stem = f"{stem}_aug_{cid}_{gen}"
                aug_items.append((aug_stem, canvas, [final_box]))
                aug_per_class[cid] += 1

        aug_stats = {
            "n_augmented_images": len(aug_items),
            "n_augmented_boxes": len(aug_items),
            "n_augmented_classes": len(aug_per_class),
        }
        print(f"[preprocess]   Hoàn tất: Đã sinh thêm {len(aug_items)} ảnh và {len(aug_items)} box cho {len(aug_per_class)} lớp trong tập Train.")

    # ── BƯỚC 5: Resize ảnh về 640×640 (Letterbox) ──
    print("[preprocess] ========================================================")
    print(f"[preprocess] BƯỚC 5: Resize toàn bộ ảnh về {image_size[0]}×{image_size[1]} (Letterbox, pad={pad_value}) & xuất data.yaml, meta.json...")

    # Xoá sạch thư mục đích để tránh file cũ còn sót lại
    for sub in ("images", "labels"):
        d = out_root / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        for sname in ("train", "val", "test"):
            (d / sname).mkdir(parents=True, exist_ok=True)

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
    for sname, stems in splits.items():
        dst_img_dir = out_root / "images" / sname
        dst_lbl_dir = out_root / "labels" / sname

        n_boxes_out = 0

        for stem in stems:
            src_img_p = stem_to_imgpath[stem]
            img = cv2.imread(str(src_img_p))
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]
            canvas, scale, dx, dy = _letterbox_image(img, image_size, pad_value)

            adj_boxes = _adjust_boxes(
                clean_boxes[stem], orig_w, orig_h, scale, dx, dy, image_size[0], image_size[1]
            )
            n_boxes_out += len(adj_boxes)

            cv2.imwrite(str(dst_img_dir / f"{stem}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            _write_labels(dst_lbl_dir / f"{stem}.txt", adj_boxes)

        meta["splits"][sname] = {
            "n_images": len(stems),
            "n_boxes_out": n_boxes_out,
        }
        total_kept += n_boxes_out
        print(f"[preprocess]   {sname}: {len(stems)} ảnh, {n_boxes_out} box")

    # Ghi toàn bộ ảnh và nhãn augmented vào tập Train
    if aug_items:
        train_img_dir = out_root / "images" / "train"
        train_lbl_dir = out_root / "labels" / "train"
        for aug_stem, canvas, aug_boxes in aug_items:
            cv2.imwrite(str(train_img_dir / f"{aug_stem}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            _write_labels(train_lbl_dir / f"{aug_stem}.txt", aug_boxes)

        meta["splits"]["train"]["n_images"] += len(aug_items)
        meta["splits"]["train"]["n_boxes_out"] += len(aug_items)
        meta["minority_augmentation"] = aug_stats
        total_kept += len(aug_items)
        print(f"[preprocess]   train (+augmented): {meta['splits']['train']['n_images']} ảnh tổng cộng ({len(aug_items)} ảnh augmented).")

    elapsed = time.time() - t0
    meta["total_dropped_tiny"] = total_dropped_tiny
    meta["total_boxes_kept"] = total_kept
    meta["elapsed_seconds"] = round(elapsed, 2)

    # Ghi data.yaml và meta.json
    _write_data_yaml(out_root, classes)
    print(f"[preprocess] data.yaml ghi tại: {out_root / 'data.yaml'}")

    meta_path = out_root / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[preprocess] meta.json ghi tại: {meta_path}")
    print(f"[preprocess] Hoàn thành toàn bộ quy trình tiền xử lý trong {elapsed:.1f}s  →  {out_root}")

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
    raw_phash_thresh = split_cfg.get("phash_threshold", 10)
    p_thresh = int(raw_phash_thresh) if (raw_phash_thresh is not None and str(raw_phash_thresh).strip().lower() not in ("none", "false")) else None

    d_root = data_root or prep.get("raw_dir", "data/raw")
    out_root = prep.get("processed_dir", "data/processed")
    target_size = tuple(prep.get("image_size", [640, 640]))
    min_box_px = int(prep.get("min_box_px", 8))
    pad_val = int(prep.get("pad_value", 114))
    aug_min = bool(prep.get("augment_minority", cfg.get("augment_minority", True)))
    raw_aug_target = prep.get("minority_target_boxes", cfg.get("minority_target_boxes", "adaptive"))
    if raw_aug_target is None or str(raw_aug_target).strip().lower() in ("adaptive", "ratio", "tier", "multiplier"):
        aug_target: int | str = "adaptive"
    elif str(raw_aug_target).strip().lower() in ("max", "-1", "none"):
        aug_target = "max"
    else:
        try:
            aug_target = int(raw_aug_target)
        except (ValueError, TypeError):
            aug_target = "adaptive"
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
        phash_thresh=p_thresh,
        seed=seed,
    )
