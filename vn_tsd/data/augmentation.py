"""Module tăng cường dữ liệu (Data Augmentation) cho các lớp trong tập Train.

Cơ chế thực hiện:
  - Cắt các biển báo và khung cảnh của class đó không lẫn các class khác
    (xác định biên an toàn loại trừ 100% mọi bounding box của các class khác trong ảnh).
  - Áp dụng biến đổi ngẫu nhiên nhưng theo seed cố định:
    + Làm mờ (Gaussian Blur / Motion Blur)
    + Xoay nhiều góc (Multi-angle rotation: -25° đến +25°)
    + Zoom in / Zoom out (scale 0.75 đến 1.35)
    + Tinh chỉnh độ sáng / tương phản nhẹ
  - Tăng cường dữ liệu thích ứng (adaptive multiplier policy):
    + Class ít mẫu (< 30 box): Tăng khoảng 2–3 lần (mặc định ~2.5x).
    + Class trung bình (30–150 box): Tăng khoảng 1.5–2 lần (mặc định ~1.75x).
    + Class nhiều mẫu (> 150 box): Giữ nguyên (1.0x).
  - Chọn ngẫu nhiên ảnh trong tập ứng viên đa dạng của class trong Train để áp dụng phép biến đổi,
    đảm bảo phân bổ đều qua các ảnh khác nhau thay vì lặp lại trên 1 ảnh.
  - Đặt vào canvas kích thước chuẩn (mặc định 640x640) với letterbox padding.
  - Tự động lọc bỏ các bounding box siêu nhỏ (< min_box_px, mặc định 8px).
  - Kiểm tra đối chiếu mã dHash với toàn bộ ảnh Validation và Test để ngăn chặn rò rỉ dữ liệu.
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any
from collections import defaultdict, Counter
import cv2
import numpy as np


def _read_raw_labels(p: Path) -> list[tuple[int, float, float, float, float]]:
    """Đọc nhãn YOLO [class_id, cx, cy, bw, bh]."""
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


def get_safe_crop_bounds(
    target_box: tuple[int, float, float, float, float],
    other_boxes: list[tuple[int, float, float, float, float]],
    W: int = 640,
    H: int = 640,
) -> tuple[float, float, float, float] | None:
    """
    Xác định giới hạn an toàn [safe_x1, safe_y1, safe_x2, safe_y2] quanh target_box
    sao cho vùng cắt hoàn toàn KHÔNG giao cắt và KHÔNG chứa bất kỳ bounding box
    nào của các class khác ('không lẫn các class khác').
    """
    t_cid, t_cx, t_cy, t_bw, t_bh = target_box
    bx1 = (t_cx - t_bw / 2.0) * W
    by1 = (t_cy - t_bh / 2.0) * H
    bx2 = (t_cx + t_bw / 2.0) * W
    by2 = (t_cy + t_bh / 2.0) * H

    safe_x1, safe_y1 = 0.0, 0.0
    safe_x2, safe_y2 = float(W), float(H)

    for o_cid, o_cx, o_cy, o_bw, o_bh in other_boxes:
        ox1 = (o_cx - o_bw / 2.0) * W
        oy1 = (o_cy - o_bh / 2.0) * H
        ox2 = (o_cx + o_bw / 2.0) * W
        oy2 = (o_cy + o_bh / 2.0) * H

        # Giao cắt phần thân trong (interior overlap)
        h_overlap = (bx1 < ox2 - 1e-3) and (bx2 > ox1 + 1e-3)
        v_overlap = (by1 < oy2 - 1e-3) and (by2 > oy1 + 1e-3)

        if h_overlap and v_overlap:
            # Hai box đè lên nhau, không thể tách sạch hoàn toàn
            return None
        elif h_overlap:
            # Tách theo phương dọc
            if by1 >= oy2 - 1e-3:
                safe_y1 = max(safe_y1, oy2)
            elif by2 <= oy1 + 1e-3:
                safe_y2 = min(safe_y2, oy1)
        elif v_overlap:
            # Tách theo phương ngang
            if bx1 >= ox2 - 1e-3:
                safe_x1 = max(safe_x1, ox2)
            elif bx2 <= ox1 + 1e-3:
                safe_x2 = min(safe_x2, ox1)
        else:
            # Hai box nằm chéo nhau, ưu tiên trục có khoảng cách lớn hơn
            dx = (bx1 - ox2) if bx1 >= ox2 else (ox1 - bx2)
            dy = (by1 - oy2) if by1 >= oy2 else (oy1 - by2)
            if dx >= dy:
                if bx1 >= ox2:
                    safe_x1 = max(safe_x1, ox2)
                else:
                    safe_x2 = min(safe_x2, ox1)
            else:
                if by1 >= oy2:
                    safe_y1 = max(safe_y1, oy2)
                else:
                    safe_y2 = min(safe_y2, oy1)

    if safe_x1 <= bx1 + 1e-3 and bx2 - 1e-3 <= safe_x2 and safe_y1 <= by1 + 1e-3 and by2 - 1e-3 <= safe_y2:
        return (safe_x1, safe_y1, safe_x2, safe_y2)
    return None


def transform_and_letterbox_crop(
    img: np.ndarray,
    target_box: tuple[int, float, float, float, float],
    safe_bounds: tuple[float, float, float, float],
    rng: random.Random,
    target_size: tuple[int, int] = (640, 640),
    min_box_px: int = 8,
    pad_value: int = 114,
) -> tuple[np.ndarray, tuple[int, float, float, float, float]] | tuple[None, None]:
    """
    1. Cắt biển báo cùng khung cảnh xung quanh trong giới hạn an toàn.
    2. Biến đổi ngẫu nhiên theo seed: xoay nhiều góc, zoom in / zoom out, làm mờ, chỉnh sáng nhẹ.
    3. Đưa vào canvas target_size với letterbox và tính toạ độ bounding box chuẩn.
    """
    H, W = img.shape[:2]
    cid, t_cx, t_cy, t_bw, t_bh = target_box
    bx1 = (t_cx - t_bw / 2.0) * W
    by1 = (t_cy - t_bh / 2.0) * H
    bx2 = (t_cx + t_bw / 2.0) * W
    by2 = (t_cy + t_bh / 2.0) * H
    bw = bx2 - bx1
    bh = by2 - by1

    safe_x1, safe_y1, safe_x2, safe_y2 = safe_bounds

    # Mở rộng ngẫu nhiên vùng khung cảnh xung quanh biển báo (trong safe_bounds)
    pad_factor_x = rng.uniform(0.6, 2.5)
    pad_factor_y = rng.uniform(0.6, 2.5)

    crop_x1 = int(max(safe_x1, bx1 - bw * pad_factor_x))
    crop_y1 = int(max(safe_y1, by1 - bh * pad_factor_y))
    crop_x2 = int(min(safe_x2, bx2 + bw * pad_factor_x))
    crop_y2 = int(min(safe_y2, by2 + bh * pad_factor_y))

    if crop_x2 <= crop_x1 + 10 or crop_y2 <= crop_y1 + 10:
        return None, None

    crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    ch, cw = crop.shape[:2]

    # Toạ độ box tương đối bên trong ảnh crop
    cbx1 = bx1 - crop_x1
    cby1 = by1 - crop_y1
    cbx2 = bx2 - crop_x1
    cby2 = by2 - crop_y1

    # ── Biến đổi hình học: Zoom in / Zoom out (bỏ xoay ảnh) ──
    # Bỏ xoay ảnh: giữ nguyên góc đứng của biển báo (angle = 0.0)
    angle = 0.0
    # Zoom in / Zoom out: tỉ lệ co dãn 0.75 - 1.35
    zoom = rng.uniform(0.75, 1.35)

    sign_center_x = (cbx1 + cbx2) / 2.0
    sign_center_y = (cby1 + cby2) / 2.0

    M = cv2.getRotationMatrix2D((sign_center_x, sign_center_y), angle, zoom)
    transformed_crop = cv2.warpAffine(
        crop, M, (cw, ch),
        borderMode=cv2.BORDER_REFLECT_101
    )

    # Tính toạ độ mới của 4 đỉnh bounding box
    corners = np.array([
        [cbx1, cby1, 1.0],
        [cbx2, cby1, 1.0],
        [cbx2, cby2, 1.0],
        [cbx1, cby2, 1.0],
    ]).T
    new_corners = M @ corners
    nbx1 = float(new_corners[0].min())
    nby1 = float(new_corners[1].min())
    nbx2 = float(new_corners[0].max())
    nby2 = float(new_corners[1].max())

    # Cắt gọn vào khung hình crop
    nbx1 = max(0.0, min(float(cw), nbx1))
    nby1 = max(0.0, min(float(ch), nby1))
    nbx2 = max(0.0, min(float(cw), nbx2))
    nby2 = max(0.0, min(float(ch), nby2))

    if (nbx2 - nbx1) < min_box_px or (nby2 - nby1) < min_box_px:
        return None, None

    # ── Biến đổi quang học: Làm mờ nhẹ (mờ nhưng vẫn nhìn thấy rõ) & Ánh sáng nhẹ ──
    # Xác suất 35% làm mờ nhẹ (kernel 3x3, sigma 0.6 - 1.0 giữ chi tiết biển báo rõ ràng)
    if rng.random() < 0.35:
        sigma = rng.uniform(0.6, 1.0)
        transformed_crop = cv2.GaussianBlur(transformed_crop, (3, 3), sigmaX=sigma, sigmaY=sigma)

    # Đổi sáng / tương phản nhẹ nhàng
    alpha = rng.uniform(0.88, 1.15)
    beta = rng.uniform(-18, 18)
    transformed_crop = cv2.convertScaleAbs(transformed_crop, alpha=alpha, beta=beta)

    # ── Letterbox lên kích thước chuẩn target_size (640, 640) ──
    tw, th = target_size
    scale = min(tw / cw, th / ch)
    rw = int(round(cw * scale))
    rh = int(round(ch * scale))
    resized_patch = cv2.resize(transformed_crop, (rw, rh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((th, tw, 3), pad_value, dtype=np.uint8)
    dx = (tw - rw) // 2
    dy = (th - rh) // 2
    canvas[dy : dy + rh, dx : dx + rw] = resized_patch

    # Toạ độ cuối cùng trên canvas (chuẩn hoá [0, 1])
    final_x1 = nbx1 * scale + dx
    final_y1 = nby1 * scale + dy
    final_x2 = nbx2 * scale + dx
    final_y2 = nby2 * scale + dy

    final_w = (final_x2 - final_x1)
    final_h = (final_y2 - final_y1)
    if final_w < min_box_px or final_h < min_box_px:
        return None, None

    final_cx = (final_x1 + final_x2) / 2.0 / tw
    final_cy = (final_y1 + final_y2) / 2.0 / th
    final_bw = final_w / tw
    final_bh = final_h / th

    final_box = (cid, final_cx, final_cy, final_bw, final_bh)
    return canvas, final_box


def augment_minority_classes(
    train_img_dir: Path,
    train_lbl_dir: Path,
    val_test_packed: np.ndarray | None = None,
    target_min_boxes: int | str | None = "adaptive",
    image_size: tuple[int, int] = (640, 640),
    min_box_px: int = 8,
    pad_value: int = 114,
    max_augmented_per_class: int = 7,
    seed: int = 42,
) -> dict[str, Any]:
    """Cân bằng dữ liệu cho các lớp trong tập Train bằng cô lập biển báo và biến đổi ảnh.
    
    Quy tắc tăng cường:
      - Class ít mẫu (< 30 box): Tăng khoảng 2-3 lần (mặc định 2.5x).
      - Class có số mẫu trung bình (30 - 150 box): Tăng khoảng 1.5-2 lần (mặc định 1.75x).
      - Class nhiều mẫu (> 150 box): Giữ nguyên (1.0x).
      - Giới hạn tối đa: không quá 7 ảnh cộng thêm cho bất kỳ class nào (max_augmented_per_class = 7).
      - Chọn ngẫu nhiên ảnh trong tập ứng viên của class để áp dụng biến đổi,
        phân bổ đều qua các ảnh khác nhau trong class trước khi lặp lại.
    """
    rng = random.Random(seed)
    if not train_img_dir.is_dir() or not train_lbl_dir.is_dir():
        return {"n_augmented_images": 0, "n_augmented_boxes": 0}

    # 1. Quét toàn bộ nhãn ảnh gốc (không tính ảnh _aug nếu chạy lại)
    base_lbl_paths = sorted([p for p in train_lbl_dir.glob("*.txt") if not ("_aug" in p.stem)])
    
    img_boxes: dict[str, list[tuple[int, float, float, float, float]]] = {}
    class_candidates: dict[int, list[tuple[str, tuple[int, float, float, float, float], tuple[float, float, float, float]]]] = defaultdict(list)
    class_counts: Counter[int] = Counter()

    for lbl_p in base_lbl_paths:
        stem = lbl_p.stem
        boxes = _read_raw_labels(lbl_p)
        if not boxes:
            continue
        img_boxes[stem] = boxes
        for i, b in enumerate(boxes):
            class_counts[b[0]] += 1
            other_b = [other for j, other in enumerate(boxes) if j != i and other[0] != b[0]]
            bounds = get_safe_crop_bounds(b, other_b, image_size[0], image_size[1])
            if bounds is not None:
                class_candidates[b[0]].append((stem, b, bounds))

    if not class_counts:
        return {"n_augmented_images": 0, "n_augmented_boxes": 0}

    # Xác định số box mục tiêu cho từng class
    class_targets: dict[int, int] = {}
    policy_str = str(target_min_boxes).strip().lower() if target_min_boxes is not None else "adaptive"

    if policy_str in ("adaptive", "ratio", "tier", "multiplier"):
        for cid, cnt in class_counts.items():
            if cnt < 30:
                # Class ít mẫu: tăng khoảng 2-3 lần (2.5x)
                target_cnt = max(int(round(cnt * 2.5)), min(15, cnt * 3))
            elif cnt <= 150:
                # Class trung bình: tăng khoảng 1.5-2 lần (1.75x)
                target_cnt = int(round(cnt * 1.75))
            else:
                # Class nhiều mẫu: giữ nguyên 1.0x
                target_cnt = cnt
            class_targets[cid] = target_cnt
    elif policy_str in ("max", "-1", "none"):
        max_count = max(class_counts.values())
        for cid in class_counts:
            class_targets[cid] = max_count
    else:
        fixed_target = int(target_min_boxes)
        for cid, cnt in class_counts.items():
            class_targets[cid] = max(cnt, fixed_target)

    # Giới hạn tối đa không quá max_augmented_per_class (mặc định 7) ảnh cộng thêm mỗi class
    class_needed = {
        c: min(class_targets[c] - class_counts[c], max_augmented_per_class)
        for c in class_counts
        if class_targets[c] > class_counts[c]
    }
    if not class_needed:
        print("[augment] Tất cả các lớp trong train đã đạt mục tiêu, không cần augment.")
        return {"n_augmented_images": 0, "n_augmented_boxes": 0}

    total_needed = sum(class_needed.values())
    print(
        f"[augment] Bắt đầu tăng cường cho {len(class_needed)} lớp trong Train (chính sách: {policy_str}, trần tối đa: {max_augmented_per_class} ảnh/lớp). "
        f"Cần sinh thêm ~{total_needed:,} box..."
    )
    print(
        "[augment]   Cơ chế: Chọn ảnh ngẫu nhiên từ tập ứng viên đa dạng -> Cắt biển báo + bối cảnh an toàn "
        "-> Zoom in/out, Làm mờ nhẹ -> Đưa về canvas 640x640."
    )

    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    # Cache ảnh gốc đã đọc để tăng tốc
    img_cache: dict[str, np.ndarray] = {}
    def get_img(st: str) -> np.ndarray | None:
        if st in img_cache:
            return img_cache[st]
        p = train_img_dir / f"{st}.jpg"
        if not p.exists():
            return None
        im = cv2.imread(str(p))
        if len(img_cache) < 250:  # Giữ tối đa 250 ảnh trong RAM
            img_cache[st] = im
        return im

    n_augmented_images = 0
    n_augmented_boxes = 0
    t_start = time.time()

    # Sắp xếp augment các class từ ít nhất đến nhiều nhất
    sorted_classes = sorted(class_needed.items(), key=lambda x: class_counts[x[0]])
    stem_aug_count: Counter[str] = Counter()

    for cid, needed in sorted_classes:
        candidates = class_candidates.get(cid, [])
        if not candidates:
            print(f"[augment]   Cảnh báo: Không tìm thấy mẫu cô lập hợp lệ cho Class {cid}, bỏ qua.")
            continue

        c_created = 0
        candidate_pool: list[tuple[str, tuple[int, float, float, float, float], tuple[float, float, float, float]]] = []
        attempts_total = 0
        max_attempts_class = max(needed * 30, 200)

        while c_created < needed and attempts_total < max_attempts_class:
            if not candidate_pool:
                # Xáo trộn ngẫu nhiên toàn bộ danh sách ảnh ứng viên khác nhau của class
                candidate_pool = list(candidates)
                rng.shuffle(candidate_pool)

            # Rút ngẫu nhiên một ảnh ứng viên khác nhau từ pool
            stem, target_b, safe_bounds = candidate_pool.pop()
            attempts_total += 1

            src_img = get_img(stem)
            if src_img is None:
                continue

            passed = False
            for attempt in range(4):
                canvas, final_box = transform_and_letterbox_crop(
                    img=src_img,
                    target_box=target_b,
                    safe_bounds=safe_bounds,
                    rng=rng,
                    target_size=image_size,
                    min_box_px=min_box_px,
                    pad_value=pad_value,
                )
                if canvas is None or final_box is None:
                    continue

                # Kiểm tra rò rỉ dữ liệu sang Val/Test bằng fast dHash (nếu có val_test_packed)
                if val_test_packed is not None:
                    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                    res = cv2.resize(gray, (33, 32), interpolation=cv2.INTER_AREA)
                    diff = res[:, 1:] > res[:, :-1]
                    cand_packed = np.packbits(diff.flatten()[None, :], axis=1)
                    xor = np.bitwise_xor(cand_packed, val_test_packed)
                    dist = popcount[xor].sum(axis=1)
                    if dist.min() <= 374:
                        continue  # Rò rỉ quang cảnh, thử biến đổi khác

                passed = True
                break

            if not passed:
                continue

            c_created += 1
            n_augmented_images += 1
            n_augmented_boxes += 1

            stem_aug_count[stem] += 1
            aug_stem = f"{stem}_c{cid}_aug{stem_aug_count[stem]}"
            dst_img = train_img_dir / f"{aug_stem}.jpg"
            dst_lbl = train_lbl_dir / f"{aug_stem}.txt"

            cv2.imwrite(str(dst_img), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
            _write_labels(dst_lbl, [final_box])

            if n_augmented_images % 1000 == 0:
                elapsed = time.time() - t_start
                rate = n_augmented_images / elapsed if elapsed > 0 else 0
                print(f"[augment]   Đã sinh {n_augmented_images:,}/{total_needed:,} ảnh ({rate:.1f} ảnh/s)...")

    elapsed_total = time.time() - t_start
    print(
        f"[augment]   Hoàn tất: Đã sinh thêm {n_augmented_images:,} ảnh và {n_augmented_boxes:,} box "
        f"cho {len(class_needed)} lớp trong tập train ({elapsed_total:.1f}s)."
    )

    return {
        "n_augmented_images": n_augmented_images,
        "n_augmented_boxes": n_augmented_boxes,
        "minority_classes_augmented": list(class_needed.keys()),
        "target_boxes": policy_str,
    }

