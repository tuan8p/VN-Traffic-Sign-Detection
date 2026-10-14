"""Offline data analysis for the detection set (boxes, not crops).

# Windows UTF-8 fix – must be before any other import that touches stdout
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

Produces figures (write under analysis/figures/):
  01_class_distribution     – số box theo từng lớp (52 lớp VNTS)
  02_box_dimensions         – phân bố w, h, tỉ lệ khung hình, diện tích
  03_box_center_heatmap     – 2D KDE vị trí tâm (cx, cy) trên khung hình
  04_images_per_split       – số ảnh và số box theo split (train/val/test)
  05_tiny_box_candidates    – histogram box kích thước < min_box_px

Run:
    python -m analysis.eda --data-root data/raw
    python -m analysis.eda --data-root data/processed  # lần 2 sau preprocess
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless, không cần display
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_classes(class_table: str | Path) -> dict[int, str]:
    """Đọc configs/classes.csv → {class_id: name_vi}."""
    table: dict[int, str] = {}
    with open(class_table, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["class_id"])
            table[cid] = row["name_vi"]
    return table


def _read_yolo_labels(
    labels_dir: Path,
    image_size_px: int | None = None,
) -> list[dict[str, Any]]:
    """Đọc tất cả file .txt YOLO trong labels_dir.

    Mỗi bounding box trả về dict:
      class_id, cx, cy, w, h  (normalized 0-1)
      w_px, h_px              (pixel, chỉ khi image_size_px không None)
      stem                    (tên file không đuôi)
    """
    boxes: list[dict[str, Any]] = []
    for txt in sorted(labels_dir.glob("*.txt")):
        stem = txt.stem
        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid, cx, cy, bw, bh = (
                    int(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
                entry: dict[str, Any] = dict(
                    class_id=cid, cx=cx, cy=cy, w=bw, h=bh, stem=stem
                )
                if image_size_px is not None:
                    entry["w_px"] = bw * image_size_px
                    entry["h_px"] = bh * image_size_px
                boxes.append(entry)
    return boxes


def _discover_splits(
    data_root: Path,
    split_names: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, Path] | None:
    """Kiểm tra cấu trúc data/processed hoặc split_dataset bên trong raw.

    Trả về dict {split: labels_dir} hoặc None nếu không thấy split.
    """
    # Cấu trúc processed: labels/{train,val,test}/
    labels_root = data_root / "labels"
    if labels_root.is_dir():
        found = {
            s: labels_root / s
            for s in split_names
            if (labels_root / s).is_dir()
        }
        if found:
            return found
    # Cấu trúc raw + split_dataset/: thư mục split_dataset/ chứa train.txt, val.txt...
    split_dir = data_root / "split_dataset"
    if split_dir.is_dir():
        txt_splits = {
            s: split_dir / f"{s}.txt"
            for s in split_names
            if (split_dir / f"{s}.txt").exists()
        }
        if txt_splits:
            return txt_splits  # giá trị là Path file .txt, xử lý riêng
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Phần 1: Class distribution
# ──────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    boxes: list[dict[str, Any]],
    classes: dict[int, str],
    out_dir: Path,
    tag: str = "",
) -> None:
    """Figure 01: số bounding box theo từng lớp, sắp xếp giảm dần."""
    num_classes = max(classes.keys()) + 1 if classes else 52
    counts = np.zeros(num_classes, dtype=int)
    for b in boxes:
        cid = b["class_id"]
        if 0 <= cid < num_classes:
            counts[cid] += 1

    order = np.argsort(counts)[::-1]
    sorted_counts = counts[order]
    sorted_labels = [
        f"{classes.get(i, str(i))}" for i in order
    ]

    # Chia 2 subplot: bên trái top-20, bên phải đầy đủ
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f"Phân bố lớp (số bounding box){' – ' + tag if tag else ''}", fontsize=14, fontweight="bold")

    # Top-20
    ax0 = axes[0]
    top_n = min(20, len(order))
    bars = ax0.barh(
        range(top_n),
        sorted_counts[:top_n],
        color=plt.cm.RdYlGn(np.linspace(0.8, 0.2, top_n)),
    )
    ax0.set_yticks(range(top_n))
    ax0.set_yticklabels([sorted_labels[i][:30] for i in range(top_n)], fontsize=7)
    ax0.invert_yaxis()
    ax0.set_xlabel("Số box")
    ax0.set_title(f"Top {top_n} lớp nhiều box nhất")
    for bar, cnt in zip(bars, sorted_counts[:top_n]):
        ax0.text(bar.get_width() + max(sorted_counts) * 0.005, bar.get_y() + bar.get_height() / 2, str(cnt), va="center", fontsize=7)

    # Tất cả lớp theo class_id
    ax1 = axes[1]
    ax1.bar(range(num_classes), counts, color="steelblue", alpha=0.7)
    ax1.set_xlabel("Class ID")
    ax1.set_ylabel("Số box")
    ax1.set_title("Phân bố theo class_id (0–51)")
    ax1.axhline(counts.mean(), color="red", linestyle="--", linewidth=1, label=f"Mean = {counts.mean():.0f}")
    ax1.legend(fontsize=9)

    plt.tight_layout()
    fname = out_dir / ("01_class_distribution" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

    # In thống kê
    total = counts.sum()
    nonzero = np.count_nonzero(counts)
    print(f"  Tổng box: {total}, lớp có box: {nonzero}/52, max: {counts.max()}, min non-zero: {counts[counts > 0].min()}")


# ──────────────────────────────────────────────────────────────────────────────
# Phần 2: Box dimensions
# ──────────────────────────────────────────────────────────────────────────────

def plot_box_dimensions(
    boxes: list[dict[str, Any]],
    out_dir: Path,
    image_size: int = 640,
    tag: str = "",
) -> None:
    """Figure 02: phân bố kích thước box (w, h, aspect ratio, area)."""
    ws = np.array([b["w"] * image_size for b in boxes], dtype=float)
    hs = np.array([b["h"] * image_size for b in boxes], dtype=float)
    areas = ws * hs
    aspects = ws / (hs + 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Phân bố kích thước bounding box{' – ' + tag if tag else ''}", fontsize=13, fontweight="bold")

    def _hist(ax, data, xlabel, title, color, bins=60):
        ax.hist(data, bins=bins, color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(np.median(data), color="red", linestyle="--", label=f"Median={np.median(data):.1f}")
        ax.axvline(np.mean(data), color="orange", linestyle=":", label=f"Mean={np.mean(data):.1f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Số box")
        ax.set_title(title)
        ax.legend(fontsize=8)

    _hist(axes[0][0], ws,      "Width (px)",  "Chiều rộng box (px)",         "steelblue")
    _hist(axes[0][1], hs,      "Height (px)", "Chiều cao box (px)",           "coral")
    _hist(axes[1][0], aspects, "w / h",       "Tỉ lệ khung hình (width/height)", "seagreen")
    _hist(axes[1][1], areas,   "Area (px²)",  "Diện tích box (px²)",          "mediumpurple")

    plt.tight_layout()
    fname = out_dir / ("02_box_dimensions" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")
    print(f"  w_px: median={np.median(ws):.1f}, mean={np.mean(ws):.1f}")
    print(f"  h_px: median={np.median(hs):.1f}, mean={np.mean(hs):.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# Phần 3: Box center heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_box_center_heatmap(
    boxes: list[dict[str, Any]],
    out_dir: Path,
    tag: str = "",
    resolution: int = 64,
) -> None:
    """Figure 03: 2D heatmap / histogram vị trí tâm (cx, cy) trên khung hình."""
    cxs = np.array([b["cx"] for b in boxes], dtype=float)
    cys = np.array([b["cy"] for b in boxes], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Heatmap vị trí tâm bounding box{' – ' + tag if tag else ''}", fontsize=13, fontweight="bold")

    # 2D histogram
    ax0 = axes[0]
    heatmap, xedges, yedges = np.histogram2d(cxs, cys, bins=resolution, range=[[0, 1], [0, 1]])
    im = ax0.imshow(
        heatmap.T,
        origin="upper",
        extent=[0, 1, 0, 1],
        cmap="hot",
        aspect="equal",
    )
    ax0.set_xlabel("cx (normalized)")
    ax0.set_ylabel("cy (normalized)")
    ax0.set_title("2D Heatmap mật độ tâm box")
    plt.colorbar(im, ax=ax0, label="Số box")

    # Scatter (subsample 3000)
    ax1 = axes[1]
    idx = np.random.choice(len(cxs), min(3000, len(cxs)), replace=False)
    ax1.scatter(cxs[idx], cys[idx], alpha=0.15, s=4, color="steelblue")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.invert_yaxis()  # (0,0) ở góc trên trái như tọa độ ảnh
    ax1.set_xlabel("cx (normalized)")
    ax1.set_ylabel("cy (normalized)")
    ax1.set_title(f"Scatter tâm box (subsample {len(idx)})")
    ax1.set_aspect("equal")

    plt.tight_layout()
    fname = out_dir / ("03_box_center_heatmap" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")
    print(f"  cx: mean={cxs.mean():.3f}, cy: mean={cys.mean():.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# Phần 4: Images per split
# ──────────────────────────────────────────────────────────────────────────────

def plot_images_per_split(
    split_box_counts: dict[str, tuple[int, int]],
    out_dir: Path,
    tag: str = "",
) -> None:
    """Figure 04: số ảnh và số box theo split.

    split_box_counts: {split_name: (n_images, n_boxes)}
    """
    splits = list(split_box_counts.keys())
    n_imgs = [split_box_counts[s][0] for s in splits]
    n_boxes = [split_box_counts[s][1] for s in splits]

    x = np.arange(len(splits))
    width = 0.35
    colors_img = ["#4C72B0", "#DD8452", "#55A868"]
    colors_box = ["#2196F3", "#FF7043", "#43A047"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Số ảnh và box theo split{' – ' + tag if tag else ''}", fontsize=13, fontweight="bold")

    bars1 = ax1.bar(x, n_imgs, color=colors_img[:len(splits)], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits, fontsize=11)
    ax1.set_ylabel("Số ảnh")
    ax1.set_title("Số ảnh theo split")
    for bar, v in zip(bars1, n_imgs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(n_imgs) * 0.01, str(v), ha="center", fontsize=10, fontweight="bold")

    bars2 = ax2.bar(x, n_boxes, color=colors_box[:len(splits)], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(splits, fontsize=11)
    ax2.set_ylabel("Số box")
    ax2.set_title("Số bounding box theo split")
    for bar, v in zip(bars2, n_boxes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(n_boxes) * 0.01, str(v), ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fname = out_dir / ("04_images_per_split" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")
    for s in splits:
        ni, nb = split_box_counts[s]
        print(f"  {s}: {ni} ảnh, {nb} box, avg box/img={nb/max(ni,1):.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# Phần 5: Tiny box candidates
# ──────────────────────────────────────────────────────────────────────────────

def plot_tiny_box_candidates(
    boxes: list[dict[str, Any]],
    out_dir: Path,
    min_box_px: int = 8,
    image_size: int = 640,
    tag: str = "",
) -> None:
    """Figure 05: histogram kích thước nhỏ; highlight box dưới min_box_px."""
    ws = np.array([b["w"] * image_size for b in boxes], dtype=float)
    hs = np.array([b["h"] * image_size for b in boxes], dtype=float)
    min_dim = np.minimum(ws, hs)  # cạnh nhỏ hơn của mỗi box

    tiny_mask = min_dim < min_box_px
    n_tiny = tiny_mask.sum()
    pct = 100 * n_tiny / max(len(boxes), 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Box nhỏ (min(w,h) < {min_box_px}px) – {n_tiny}/{len(boxes)} box ({pct:.2f}%){' – ' + tag if tag else ''}",
        fontsize=12, fontweight="bold"
    )

    # Histogram min dimension (zoom vào phần nhỏ)
    ax0 = axes[0]
    threshold_display = min_box_px * 10
    small_vals = min_dim[min_dim < threshold_display]
    if len(small_vals) > 0:
        ax0.hist(small_vals, bins=40, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.3, label="all")
        ax0.hist(small_vals[small_vals < min_box_px], bins=20, color="red", alpha=0.8, label=f"< {min_box_px}px (bị loại)")
    ax0.axvline(min_box_px, color="red", linestyle="--", linewidth=2, label=f"Ngưỡng = {min_box_px}px")
    ax0.set_xlabel("min(w, h) – pixels")
    ax0.set_ylabel("Số box")
    ax0.set_title(f"Phân bố cạnh nhỏ nhất (zoom ≤ {threshold_display}px)")
    ax0.legend(fontsize=9)

    # Scatter: w vs h, tô màu box tiny
    ax1 = axes[1]
    normal_mask = ~tiny_mask
    sample_n = min(normal_mask.sum(), 5000)
    sample_t = min(n_tiny, 1000)
    idx_n = np.random.choice(np.where(normal_mask)[0], sample_n, replace=False) if sample_n > 0 else []
    idx_t = np.random.choice(np.where(tiny_mask)[0], sample_t, replace=False) if sample_t > 0 else []
    if len(idx_n):
        ax1.scatter(ws[idx_n], hs[idx_n], alpha=0.15, s=4, color="steelblue", label="Normal")
    if len(idx_t):
        ax1.scatter(ws[idx_t], hs[idx_t], alpha=0.6, s=10, color="red", label=f"Tiny (<{min_box_px}px)")
    ax1.axhline(min_box_px, color="orange", linestyle="--", linewidth=1)
    ax1.axvline(min_box_px, color="orange", linestyle="--", linewidth=1, label=f"min_box_px={min_box_px}")
    ax1.set_xlabel("Width (px)")
    ax1.set_ylabel("Height (px)")
    ax1.set_title("Scatter w vs h (subsample)")
    ax1.legend(fontsize=9)

    plt.tight_layout()
    fname = out_dir / ("05_tiny_box_candidates" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")
    print(f"  Tiny box: {n_tiny}/{len(boxes)} ({pct:.2f}%) có min(w,h) < {min_box_px}px")


# ──────────────────────────────────────────────────────────────────────────────
# Hàm collect_data: gom toàn bộ box và thông tin split
# ──────────────────────────────────────────────────────────────────────────────

def collect_data(
    data_root: Path,
    image_size: int = 640,
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, int]], str]:
    """Đọc dữ liệu từ data_root.

    Returns:
        all_boxes   : danh sách tất cả box (mọi split gộp lại)
        split_stats : {split: (n_images, n_boxes)}
        data_tag    : tên nhận diện (raw / processed) để đặt tên file output
    """
    data_root = Path(data_root).resolve()

    # Xác định tag
    tag = data_root.name  # "raw" hoặc "processed" hoặc tên thư mục

    # Cấu trúc 1: labels/{train,val,test}/ (data/processed)
    labels_root = data_root / "labels"
    split_names = ("train", "val", "test")

    if labels_root.is_dir() and any((labels_root / s).is_dir() for s in split_names):
        print(f"  Phát hiện cấu trúc processed: labels/{{train,val,test}}/")
        all_boxes: list[dict[str, Any]] = []
        split_stats: dict[str, tuple[int, int]] = {}
        for sname in split_names:
            sdir = labels_root / sname
            if not sdir.is_dir():
                continue
            img_dir = data_root / "images" / sname
            n_imgs = len(list(img_dir.glob("*.*"))) if img_dir.is_dir() else len(list(sdir.glob("*.txt")))
            sboxes = _read_yolo_labels(sdir, image_size_px=image_size)
            all_boxes.extend(sboxes)
            split_stats[sname] = (n_imgs, len(sboxes))
        return all_boxes, split_stats, tag

    # Cấu trúc 2: labels/ phẳng + split_dataset/{train,val,test}.txt (hoặc không có split)
    flat_labels_dir = data_root / "labels"
    if not flat_labels_dir.is_dir():
        # Thử chính data_root là labels dir
        flat_labels_dir = data_root

    all_boxes = _read_yolo_labels(flat_labels_dir, image_size_px=image_size)
    stems_all: set[str] = {b["stem"] for b in all_boxes}

    # Kiểm tra split_dataset/
    split_dir = data_root / "split_dataset"
    split_stats = {}

    if split_dir.is_dir():
        for sname in split_names:
            sf = split_dir / f"{sname}.txt"
            if not sf.exists():
                continue
            with open(sf, "r", encoding="utf-8") as fp:
                lines = [l.strip() for l in fp if l.strip()]
            # Mỗi dòng là tên ảnh (có hoặc không có đuôi)
            sset = {Path(l).stem for l in lines}
            s_boxes = [b for b in all_boxes if b["stem"] in sset]
            split_stats[sname] = (len(sset), len(s_boxes))
        if split_stats:
            return all_boxes, split_stats, tag

    # Không có split: báo all
    img_dir = data_root / "images"
    n_imgs = len(list(img_dir.glob("*.*"))) if img_dir.is_dir() else len(stems_all)
    split_stats["all"] = (n_imgs, len(all_boxes))
    return all_boxes, split_stats, tag


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Offline EDA cho bộ detection VNTS (boxes, không crop)"
    )
    p.add_argument("--data-root", default=None, help="Thư mục gốc dataset (raw hoặc processed)")
    p.add_argument("--figures-dir", default=None, help="Thư mục xuất hình (mặc định: analysis/figures/)")
    p.add_argument("--class-table", default=None, help="Đường dẫn configs/classes.csv")
    p.add_argument("--image-size", type=int, default=640, help="Kích thước ảnh để tính px (default: 640)")
    p.add_argument("--min-box-px", type=int, default=8, help="Ngưỡng tiny box (default: 8)")
    args = p.parse_args(argv)

    # Resolve data root
    data_root_str = args.data_root or os.environ.get("DATA_ROOT") or ""
    if not data_root_str:
        print("[EDA] Chưa có --data-root. Dùng: python -m analysis.eda --data-root data/raw")
        sys.exit(1)
    data_root = Path(data_root_str).resolve()
    if not data_root.exists():
        print(f"[EDA] Không tìm thấy thư mục: {data_root}")
        sys.exit(1)

    # Figures dir
    _repo_root = Path(__file__).resolve().parents[1]
    figures_dir = Path(args.figures_dir) if args.figures_dir else _repo_root / "analysis" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Class table
    class_table = Path(args.class_table) if args.class_table else _repo_root / "configs" / "classes.csv"
    classes: dict[int, str] = {}
    if class_table.exists():
        classes = _load_classes(class_table)
        print(f"[EDA] Đọc {len(classes)} lớp từ {class_table}")
    else:
        print(f"[EDA] Không tìm thấy class table: {class_table}")

    # Thu thập dữ liệu
    print(f"\n[EDA] Đọc dữ liệu từ: {data_root}")
    all_boxes, split_stats, tag = collect_data(data_root, image_size=args.image_size)

    if not all_boxes:
        print("[EDA] Không tìm thấy box nào. Kiểm tra lại cấu trúc thư mục.")
        sys.exit(1)

    print(f"[EDA] Tổng: {len(all_boxes)} box, splits: {list(split_stats.keys())}")

    np.random.seed(42)  # reproducible scatter / subsample

    # ── Figure 01 ──
    print("\n[EDA] 01 Class distribution …")
    plot_class_distribution(all_boxes, classes, figures_dir, tag=tag)

    # ── Figure 02 ──
    print("\n[EDA] 02 Box dimensions …")
    plot_box_dimensions(all_boxes, figures_dir, image_size=args.image_size, tag=tag)

    # ── Figure 03 ──
    print("\n[EDA] 03 Box center heatmap …")
    plot_box_center_heatmap(all_boxes, figures_dir, tag=tag)

    # ── Figure 04 ──
    print("\n[EDA] 04 Images per split …")
    plot_images_per_split(split_stats, figures_dir, tag=tag)

    # ── Figure 05 ──
    print("\n[EDA] 05 Tiny box candidates …")
    plot_tiny_box_candidates(
        all_boxes,
        figures_dir,
        min_box_px=args.min_box_px,
        image_size=args.image_size,
        tag=tag,
    )

    print(f"\n[EDA] Xong. Hình đã ghi vào: {figures_dir}")


if __name__ == "__main__":
    main()
