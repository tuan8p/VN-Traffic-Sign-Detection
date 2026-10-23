"""Module tăng cường dữ liệu (Data Augmentation) cho các lớp thiểu số trong tập Train.

Đảm bảo:
  - Chỉ tăng cường các lớp có số lượng box < target_min_boxes (mặc định 50).
  - Tự động lọc bỏ các bounding box siêu nhỏ (< min_box_px, mặc định 8px) sinh ra do scale.
  - Kiểm tra đối chiếu mã dHash với toàn bộ ảnh trong tập Validation và Test
    với ngưỡng an toàn (<= 374 bit), loại bỏ ngay các ảnh biến đổi có nguy cơ
    gây rò rỉ dữ liệu (Data Leakage) sang các tập khác.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any
from collections import defaultdict, Counter
import cv2
import numpy as np
from PIL import Image

from analysis.image_utils import calculate_dhash


def _read_raw_labels(p: Path) -> list[tuple[int, float, float, float, float]]:
    """Đọc label YOLO [class_id, cx, cy, bw, bh]."""
    boxes = []
    if not p.exists():
        return boxes
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cid = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    boxes.append((cid, cx, cy, bw, bh))
                except ValueError:
                    continue
    return boxes


def _write_labels(p: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    """Ghi nhãn YOLO format chuẩn."""
    lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n" for b in boxes]
    with open(p, "w", encoding="utf-8") as f:
        f.writelines(lines)


def augment_minority_classes(
    train_img_dir: Path,
    train_lbl_dir: Path,
    val_test_packed: np.ndarray | None = None,
    target_min_boxes: int = 50,
    min_box_px: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Cân bằng dữ liệu cho các lớp thiểu số trong tập train bằng data augmentation."""
    rng = random.Random(seed)
    if not train_img_dir.is_dir() or not train_lbl_dir.is_dir():
        return {"n_augmented_images": 0, "n_augmented_boxes": 0}

    # Đọc nhãn của toàn bộ ảnh train gốc (loại trừ các ảnh aug nếu chạy lại)
    all_img_paths = sorted(train_img_dir.glob("*.jpg"))
    base_stems = [p.stem for p in all_img_paths if not ("_aug" in p.stem)]

    img_to_boxes: dict[str, list[tuple[int, float, float, float, float]]] = {}
    class_to_stems: dict[int, list[str]] = defaultdict(list)
    class_counts: Counter[int] = Counter()

    for stem in base_stems:
        lbl_p = train_lbl_dir / f"{stem}.txt"
        boxes = _read_raw_labels(lbl_p)
        if boxes:
            img_to_boxes[stem] = boxes
            for cid, cx, cy, bw, bh in boxes:
                class_counts[cid] += 1
                class_to_stems[cid].append(stem)

    minority_classes = {c: cnt for c, cnt in class_counts.items() if cnt < target_min_boxes}
    if not minority_classes:
        print("[augment] Tất cả các lớp trong train đã đủ số lượng, không cần augment.")
        return {"n_augmented_images": 0, "n_augmented_boxes": 0}

    print(f"[augment] Phát hiện {len(minority_classes)} lớp thiểu số (< {target_min_boxes} box). Đang tiến hành data augmentation...")

    stem_aug_needed: dict[str, int] = defaultdict(int)
    for cid, cnt in sorted(minority_classes.items(), key=lambda x: x[1]):
        needed = target_min_boxes - cnt
        stems_with_c = list(set(class_to_stems[cid]))
        if not stems_with_c:
            continue
        num_per_img = max(1, min(6, round(needed / len(stems_with_c))))
        for stem in stems_with_c:
            stem_aug_needed[stem] = max(stem_aug_needed[stem], num_per_img)

    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    n_augmented_images = 0
    n_augmented_boxes = 0

    for stem, n_augs in stem_aug_needed.items():
        src_img_p = train_img_dir / f"{stem}.jpg"
        if not src_img_p.exists():
            continue
        img = cv2.imread(str(src_img_p))
        if img is None:
            continue
        boxes = img_to_boxes.get(stem, [])
        if not boxes:
            continue

        h, w = img.shape[:2]

        for k in range(1, n_augs + 1):
            passed = False
            for attempt in range(4):
                aug_img = img.copy()

                # 1. Biến đổi ánh sáng / tương phản
                alpha = rng.uniform(0.85, 1.20)
                beta = rng.uniform(-22, 22)
                aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

                # 2. HSV color jitter
                hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
                hue_shift = rng.uniform(-6, 6)
                sat_scale = rng.uniform(0.82, 1.18)
                val_scale = rng.uniform(0.85, 1.15)
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_scale, 0, 255)
                aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

                # 3. Làm mờ nhẹ ngẫu nhiên (30% xác suất)
                if rng.random() < 0.30:
                    ksize = rng.choice([3, 5])
                    aug_img = cv2.GaussianBlur(aug_img, (ksize, ksize), 0)

                # 4. Co dãn / dịch chuyển nhẹ (20% xác suất)
                aug_boxes = []
                if rng.random() < 0.20:
                    scale = rng.uniform(0.94, 1.04)
                    max_dx = int(w * 0.03)
                    max_dy = int(h * 0.03)
                    dx = rng.randint(-max_dx, max_dx)
                    dy = rng.randint(-max_dy, max_dy)
                    M = np.float32([[scale, 0, (1 - scale) * w / 2 + dx], [0, scale, (1 - scale) * h / 2 + dy]])
                    aug_img = cv2.warpAffine(aug_img, M, (w, h), borderValue=(114, 114, 114))

                    for cid, cx, cy, bw, bh in boxes:
                        ncx = (cx * w * scale + (1 - scale) * w / 2 + dx) / w
                        ncy = (cy * h * scale + (1 - scale) * h / 2 + dy) / h
                        nbw = bw * scale
                        nbh = bh * scale
                        if 0 < ncx < 1 and 0 < ncy < 1 and nbw > 0 and nbh > 0:
                            x1 = max(0.0, ncx - nbw / 2)
                            y1 = max(0.0, ncy - nbh / 2)
                            x2 = min(1.0, ncx + nbw / 2)
                            y2 = min(1.0, ncy + nbh / 2)
                            if x2 > x1 and y2 > y1:
                                box_w = (x2 - x1) * w
                                box_h = (y2 - y1) * h
                                if box_w >= min_box_px and box_h >= min_box_px:
                                    aug_boxes.append((cid, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
                else:
                    aug_boxes = list(boxes)

                if not aug_boxes:
                    continue

                # 5. Kiểm tra mã dHash không rò rỉ sang Val và Test (ngưỡng an toàn 374)
                if val_test_packed is not None:
                    h_aug = calculate_dhash(aug_img, hash_size=32, strip_padding=True)
                    if h_aug is not None:
                        cand_packed = np.packbits(h_aug[None, :], axis=1)
                        xor = np.bitwise_xor(cand_packed, val_test_packed)
                        dist = popcount[xor].sum(axis=1)
                        if dist.min() <= 374:
                            continue  # Rò rỉ bối cảnh, thử lại biến đổi khác

                passed = True
                break

            if not passed or not aug_boxes:
                continue

            aug_stem = f"{stem}_aug{k}"
            dst_img = train_img_dir / f"{aug_stem}.jpg"
            dst_lbl = train_lbl_dir / f"{aug_stem}.txt"

            cv2.imwrite(str(dst_img), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            _write_labels(dst_lbl, aug_boxes)

            n_augmented_images += 1
            n_augmented_boxes += len(aug_boxes)

    print(
        f"[augment]   Đã sinh thêm {n_augmented_images} ảnh và {n_augmented_boxes} box augmented "
        f"cho {len(minority_classes)} lớp thiểu số trong tập train."
    )
    return {
        "n_augmented_images": n_augmented_images,
        "n_augmented_boxes": n_augmented_boxes,
        "minority_classes_augmented": list(minority_classes.keys()),
        "target_min_boxes": target_min_boxes,
    }
