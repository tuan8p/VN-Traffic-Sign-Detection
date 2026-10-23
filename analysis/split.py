"""Module phân chia tập dữ liệu (Dataset Splitting) dựa trên cụm dHash.

Quy trình chuẩn:
  1. Gom cụm toàn bộ ảnh đầu vào bằng dHash 1024-bit (size=32, Hamming <= 370).
  2. Bóc tách viền xám (114) để bảo đảm việc gom cụm hoàn toàn dựa trên bối cảnh thực.
  3. Phân chia các cụm vào 3 tập (Train, Val, Test) theo tỉ lệ (mặc định 70/15/15).
  4. ĐẢM BẢO TUYỆT ĐỐI:
     - Toàn bộ các ảnh thuộc cùng 1 cụm phải nằm trong cùng 1 tập (không được chia rẽ cụm).
     - Đầy đủ 52/52 lớp trong tập Train.
     - Cân bằng quang học: phân bổ đều ảnh tối và ảnh mờ vào các tập, tránh lệch dữ liệu.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any
from collections import defaultdict, Counter
import numpy as np

from analysis.image_utils import calculate_brightness_sharpness
from analysis.clustering import cluster_images_dhash


def _read_labels_for_stems(lbl_dir: Path, stems: list[str]) -> dict[str, list[int]]:
    """Đọc danh sách class_id cho từng file nhãn."""
    stem_to_classes: dict[str, list[int]] = defaultdict(list)
    for s in stems:
        p = lbl_dir / f"{s}.txt"
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        stem_to_classes[s].append(int(parts[0]))
        except Exception:
            continue
    return stem_to_classes


def split_by_dhash_clusters(
    img_paths: list[Path],
    lbl_dir: Path,
    ratio: tuple[float, float, float] = (0.70, 0.15, 0.15),
    hash_size: int = 32,
    hamming_thresh: int = 370,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Phân chia split hoàn toàn tự động dựa trên kết quả gom cụm dHash từ ảnh."""
    if not img_paths:
        return {"train": [], "val": [], "test": []}

    print(f"[split] Bước 1: Gom cụm {len(img_paths):,} ảnh bằng dHash 1024-bit (size={hash_size}, Hamming ≤ {hamming_thresh})...")
    cluster_res = cluster_images_dhash(img_paths, hash_size=hash_size, hamming_thresh=hamming_thresh)
    clusters = cluster_res.get("clusters", {})
    if not clusters:
        print("[split] Cảnh báo: Không thể gom cụm, dùng danh sách ảnh phẳng.")
        stems = [p.stem for p in img_paths]
        n = len(stems)
        n_tr = int(n * ratio[0])
        n_va = int(n * ratio[1])
        return {
            "train": stems[:n_tr],
            "val": stems[n_tr : n_tr + n_va],
            "test": stems[n_tr + n_va :],
        }

    total_images = len(img_paths)
    print(f"[split]   Tìm thấy {len(clusters):,} cụm từ {total_images:,} ảnh.")

    all_stems = [p.stem for p in img_paths]
    stem_to_classes = _read_labels_for_stems(lbl_dir, all_stems)

    # Đo kiểm quang học nhanh để cân bằng ảnh tối & mờ
    print("[split] Bước 2: Khảo sát độ sáng và độ mờ để cân bằng quang học...")
    stem_to_path = {p.stem: p for p in img_paths}
    stem_quality: dict[str, dict[str, bool]] = {}
    for s in all_stems:
        p = stem_to_path.get(s)
        if not p:
            continue
        res = calculate_brightness_sharpness(p)
        if res is not None:
            b, s_val = res
            stem_quality[s] = {
                "dark": b < 85,
                "blur": s_val < 100,
            }
        else:
            stem_quality[s] = {"dark": False, "blur": False}

    cluster_to_classes: dict[str, list[int]] = defaultdict(list)
    cluster_quality: dict[str, dict[str, int]] = defaultdict(lambda: {"dark": 0, "blur": 0, "total": 0})

    for cname, c_stems in clusters.items():
        for s in c_stems:
            cluster_to_classes[cname].extend(stem_to_classes.get(s, []))
            q = stem_quality.get(s, {})
            if q.get("dark"):
                cluster_quality[cname]["dark"] += 1
            if q.get("blur"):
                cluster_quality[cname]["blur"] += 1
            cluster_quality[cname]["total"] += 1

    tot_dark = sum(cluster_quality[c]["dark"] for c in clusters)
    tot_blur = sum(cluster_quality[c]["blur"] for c in clusters)

    target_counts = {
        "train": int(total_images * ratio[0]),
        "val": int(total_images * ratio[1]),
        "test": total_images - int(total_images * ratio[0]) - int(total_images * ratio[1]),
    }

    # Tìm các cụm bắt buộc vào train (lớp chỉ xuất hiện ở 1 cụm duy nhất, hoặc cụm quá lớn)
    class_to_clusters: dict[int, set[str]] = defaultdict(set)
    for cname, cls_list in cluster_to_classes.items():
        for cid in cls_list:
            class_to_clusters[cid].add(cname)

    mandatory_train: set[str] = set()
    for cid, cset in class_to_clusters.items():
        if len(cset) == 1:
            mandatory_train.update(cset)

    for cname, c_stems in clusters.items():
        if len(c_stems) >= 300:
            mandatory_train.add(cname)

    print(f"[split]   Cụm bắt buộc vào Train (bảo toàn 52 lớp): {len(mandatory_train)} cụm ({sum(len(clusters[c]) for c in mandatory_train):,} ảnh).")

    # Thuật toán tìm kiếm nghiệm tối ưu cân bằng tỉ lệ, lớp và quang học
    print("[split] Bước 3: Phân bổ các cụm vào Train, Val, Test theo tỉ lệ và cô lập 100% cụm...")
    best_score = float("inf")
    best_splits: dict[str, list[str]] | None = None
    best_info: Any = None

    rng_seed = seed
    for trial in range(3000):
        rng = random.Random(rng_seed + trial)
        splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        split_sizes = {"train": 0, "val": 0, "test": 0}
        split_classes: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
        split_dark = {"train": 0, "val": 0, "test": 0}
        split_blur = {"train": 0, "val": 0, "test": 0}

        for cname in mandatory_train:
            c_stems = clusters[cname]
            splits["train"].extend(c_stems)
            split_sizes["train"] += len(c_stems)
            split_classes["train"].update(cluster_to_classes[cname])
            split_dark["train"] += cluster_quality[cname]["dark"]
            split_blur["train"] += cluster_quality[cname]["blur"]

        remaining = [c for c in clusters if c not in mandatory_train]
        rng.shuffle(remaining)

        for cname in remaining:
            c_stems = clusters[cname]
            c_sz = len(c_stems)
            c_cls = set(cluster_to_classes[cname])
            c_dk = cluster_quality[cname]["dark"]
            c_bl = cluster_quality[cname]["blur"]

            missing_train = c_cls - split_classes["train"]
            if missing_train and split_sizes["train"] < target_counts["train"] + 30:
                chosen = "train"
            else:
                scores = {}
                for s in ("val", "test", "train"):
                    fill = (split_sizes[s] + c_sz) / target_counts[s]
                    new_c = len(c_cls - split_classes[s])
                    dark_fill = (split_dark[s] + c_dk) / max(1, tot_dark * (ratio[0] if s == "train" else ratio[1]))
                    blur_fill = (split_blur[s] + c_bl) / max(1, tot_blur * (ratio[0] if s == "train" else ratio[1]))
                    scores[s] = fill + 0.35 * dark_fill + 0.35 * blur_fill - 0.2 * new_c + rng.uniform(-0.02, 0.02)
                chosen = min(scores.keys(), key=lambda s: scores[s])

            splits[chosen].extend(c_stems)
            split_sizes[chosen] += c_sz
            split_classes[chosen].update(c_cls)
            split_dark[chosen] += c_dk
            split_blur[chosen] += c_bl

        if len(split_classes["train"]) < 52:
            continue

        err_size = sum(abs(split_sizes[s] - target_counts[s]) / target_counts[s] for s in splits)
        err_dark = sum(abs(split_dark[s] / max(1, tot_dark) - (ratio[0] if s == "train" else ratio[1])) for s in splits)
        err_blur = sum(abs(split_blur[s] / max(1, tot_blur) - (ratio[0] if s == "train" else ratio[1])) for s in splits)
        cov = len(split_classes["val"]) + len(split_classes["test"])

        score = err_size * 15 + err_dark * 20 + err_blur * 20 - cov
        if score < best_score:
            best_score = score
            best_splits = {s: sorted(splits[s]) for s in splits}
            best_info = (split_sizes, split_classes, split_dark, split_blur)

    if best_splits is None:
        raise RuntimeError("Không tìm thấy cấu hình split thỏa mãn điều kiện 52 lớp trong train.")

    sizes, classes, s_dark, s_blur = best_info
    print(f"[split] Hoàn thành phân chia split theo cụm (Cô lập cụm 100%):")
    print(f"[split]   Train: {sizes['train']:,} ảnh ({sizes['train']/total_images*100:.1f}%) | {len(classes['train'])}/52 lớp | {s_dark['train']} ảnh tối | {s_blur['train']} ảnh mờ")
    print(f"[split]   Val  : {sizes['val']:,} ảnh ({sizes['val']/total_images*100:.1f}%) | {len(classes['val'])}/52 lớp | {s_dark['val']} ảnh tối | {s_blur['val']} ảnh mờ")
    print(f"[split]   Test : {sizes['test']:,} ảnh ({sizes['test']/total_images*100:.1f}%) | {len(classes['test'])}/52 lớp | {s_dark['test']} ảnh tối | {s_blur['test']} ảnh mờ")

    return best_splits
