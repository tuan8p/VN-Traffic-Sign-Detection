from __future__ import annotations

import sys
import io

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""Offline data analysis for the detection set (boxes, not crops).

Bộ biểu đồ chuẩn (lưu tại analysis/figures/{raw,processed}/):
  • data/raw (8 biểu đồ trên TOÀN BỘ ảnh ban đầu, không split):
      01_class_distribution  – toàn bộ 52 lớp sắp xếp giảm dần
      02_boxes_per_image      – phân bố số lượng bounding box trên mỗi file nhãn / ảnh
      03_box_dimensions       – phân bố w, h, tỉ lệ khung hình (w/h), diện tích box
      04_box_center_heatmap   – mật độ vị trí tâm (cx, cy) trên khung hình (2D KDE & scatter)
      05_dhash_clusters       – gom cụm dHash 1024-bit (size=32, Hamming <= 370)
      06_tiny_box_candidates  – histogram và scatter box siêu nhỏ (< min_box_px)
      07_brightness_sharpness – phân tích độ sáng & độ sắc nét quang học
      08_sample_labels        – demo n ảnh mẫu vẽ bounding box và nhãn tiếng Việt

  • data/processed (9 biểu đồ sau tiền xử lý & phân chia split mới):
      01_class_distribution  – phân bố 52 lớp trên 3 subplot (Train Real vs Aug, Val, Test)
      02_boxes_per_image      – phân bố box/ảnh sau letterbox
      03_box_dimensions       – kích thước box trên canvas 640×640
      04_box_center_heatmap   – mật độ tâm box trên canvas 640×640
      05_images_per_split     – số lượng ảnh và box theo từng split mới
      06_tiny_box_candidates  – xác nhận 0% box nhỏ < 8px
      07_data_leakage         – ma trận trùng lặp dHash & khoảng cách Hamming (bỏ viền xám 114)
      08_brightness_sharpness – đo kiểm độ sáng và độ nét sau tiền xử lý
      09_sample_labels        – demo n ảnh mẫu sau tiền xử lý
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import cv2

from analysis.image_utils import (
    crop_active_region,
    calculate_dhash,
    calculate_brightness_sharpness,
)
from analysis.class_distribution import plot_class_distribution
from analysis.clustering import cluster_images_dhash, plot_dhash_clusters
from analysis.leakage import check_data_leakage, plot_data_leakage

# Cấu hình font tiếng Việt trong Matplotlib
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "Tahoma", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

__all__ = [
    "analyze_processed_dataset",
    "analyze_raw_dataset",
    "main",
]


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
    """Đọc tất cả file .txt YOLO trong labels_dir."""
    boxes: list[dict[str, Any]] = []
    if not labels_dir.is_dir():
        return boxes

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
                try:
                    cid, cx, cy, bw, bh = (
                        int(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    )
                except ValueError:
                    continue
                entry: dict[str, Any] = dict(
                    class_id=cid, cx=cx, cy=cy, w=bw, h=bh, stem=stem
                )
                if image_size_px is not None:
                    entry["w_px"] = bw * image_size_px
                    entry["h_px"] = bh * image_size_px
                boxes.append(entry)
    return boxes



def _get_image_paths(data_root: Path) -> list[Path]:
    """Tìm toàn bộ file ảnh trong data_root."""
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_dir = data_root / "images"
    search_dir = img_dir if img_dir.is_dir() else data_root

    paths = [p for p in search_dir.rglob("*.*") if p.suffix.lower() in img_extensions]
    return sorted(paths)


# ──────────────────────────────────────────────────────────────────────────────
# 02. Boxes per Image / Label File
# ──────────────────────────────────────────────────────────────────────────────

def plot_boxes_per_image(
    data_root: Path,
    boxes: list[dict[str, Any]],
    out_dir: Path,
    tag: str = "",
) -> None:
    """Figure 02: Phân bố số lượng bounding box trên mỗi file nhãn / ảnh."""
    counts = Counter(b["stem"] for b in boxes)
    values = list(counts.values()) if counts else [0]
    total_imgs = len(counts)
    mean_v = float(np.mean(values))
    median_v = float(np.median(values))
    max_v = max(values) if values else 0

    val_counts = Counter(values)
    xs = sorted(val_counts.keys())
    ys = [val_counts[x] for x in xs]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(xs, ys, color="#2B5B84", edgecolor="#1D3557", width=0.65, alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xlabel("Số lượng Bounding Box trên một ảnh", fontsize=11)
    ax.set_ylabel("Số lượng ảnh / file nhãn", fontsize=11)
    title_suffix = f" – {tag.upper()}" if tag else ""
    ax.set_title(
        f"02. Phân bố Số lượng Bounding Box trên mỗi Ảnh{title_suffix}\n"
        f"(Tổng ảnh có nhãn: {total_imgs:,} | Trung bình: {mean_v:.2f} | Trung vị: {median_v:.0f} | Max: {max_v})",
        fontsize=12.5,
        fontweight="bold",
        pad=10,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for bar, y in zip(bars, ys):
        pct = y / max(total_imgs, 1) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ys) * 0.015,
                f"{y:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    fname = out_dir / "02_boxes_per_image.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    print(f"  Boxes/image: trung bình {mean_v:.2f}, trung vị {median_v:.0f}, max {max_v}")


# ──────────────────────────────────────────────────────────────────────────────
# 03. Box Dimensions
# ──────────────────────────────────────────────────────────────────────────────

def plot_box_dimensions(
    boxes: list[dict[str, Any]],
    out_dir: Path,
    image_size: int = 640,
    tag: str = "",
) -> None:
    """Figure 03: phân bố width, height, aspect ratio và area."""
    canvas_h = int(round(image_size * 9 / 16)) if tag == "raw" else image_size
    ws_px = np.array([b["w"] * image_size for b in boxes], dtype=float)
    hs_px = np.array([b["h"] * canvas_h for b in boxes], dtype=float)
    ratios = np.where(hs_px > 0, ws_px / hs_px, 1.0)
    areas_px = ws_px * hs_px

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title_suffix = f" – {tag.upper()}" if tag else ""
    fig.suptitle(f"03. Phân bố Kích thước & Tỉ lệ Bounding Box{title_suffix}", fontsize=14, fontweight="bold")

    # Width
    axes[0, 0].hist(ws_px, bins=50, color="#457B9D", edgecolor="white", alpha=0.9)
    axes[0, 0].axvline(np.median(ws_px), color="#E63946", linestyle="--", label=f"Median = {np.median(ws_px):.1f}px")
    axes[0, 0].set_title(f"Chiều rộng (Width) theo Canvas {image_size}px", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Pixels")
    axes[0, 0].set_ylabel("Số lượng box")
    axes[0, 0].legend()
    axes[0, 0].grid(linestyle=":", alpha=0.5)

    # Height
    axes[0, 1].hist(hs_px, bins=50, color="#2A9D8F", edgecolor="white", alpha=0.9)
    axes[0, 1].axvline(np.median(hs_px), color="#E63946", linestyle="--", label=f"Median = {np.median(hs_px):.1f}px")
    axes[0, 1].set_title(f"Chiều cao (Height) theo Canvas {canvas_h}px", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Pixels")
    axes[0, 1].legend()
    axes[0, 1].grid(linestyle=":", alpha=0.5)

    # Aspect Ratio
    axes[1, 0].hist(ratios, bins=50, range=(0, 3), color="#E76F51", edgecolor="white", alpha=0.9)
    axes[1, 0].axvline(1.0, color="#1D3557", linestyle="-", label="Vuông (w/h = 1.0)")
    axes[1, 0].axvline(np.median(ratios), color="#E63946", linestyle="--", label=f"Median = {np.median(ratios):.2f}")
    axes[1, 0].set_title("Tỉ lệ Khung hình (Aspect Ratio = w / h)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Tỉ lệ (w / h)")
    axes[1, 0].set_ylabel("Số lượng box")
    axes[1, 0].legend()
    axes[1, 0].grid(linestyle=":", alpha=0.5)

    # Area
    axes[1, 1].hist(np.sqrt(areas_px), bins=50, color="#7209B7", edgecolor="white", alpha=0.9)
    axes[1, 1].axvline(np.median(np.sqrt(areas_px)), color="#E63946", linestyle="--",
                       label=f"Median = {np.median(np.sqrt(areas_px)):.1f}px")
    axes[1, 1].set_title("Căn bậc hai diện tích (sqrt(w × h))", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Pixels")
    axes[1, 1].legend()
    axes[1, 1].grid(linestyle=":", alpha=0.5)

    fig.tight_layout()
    fname = out_dir / "03_box_dimensions.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    print(f"  w_px: median={np.median(ws_px):.1f}, mean={np.mean(ws_px):.1f}")
    print(f"  h_px: median={np.median(hs_px):.1f}, mean={np.mean(hs_px):.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# 04. Box Center Heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_box_center_heatmap(
    boxes: list[dict[str, Any]],
    out_dir: Path,
    tag: str = "",
) -> None:
    """Figure 04: Heatmap 2D vị trí tâm box (cx, cy) trên khung hình chuẩn hoá [0, 1]."""
    cxs = np.array([b["cx"] for b in boxes], dtype=float)
    cys = np.array([b["cy"] for b in boxes], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 8))
    counts, xedges, yedges, img = ax.hist2d(
        cxs, cys, bins=40, range=[[0, 1], [0, 1]], cmap="YlOrRd", cmin=1
    )
    fig.colorbar(img, ax=ax, label="Số lượng Box trong ô lưới")

    ax.invert_yaxis()
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)

    title_suffix = f" – {tag.upper()}" if tag else ""
    ax.set_title(
        f"04. Heatmap Mật độ Vị trí Tâm Bounding Box (cx, cy){title_suffix}\n"
        f"(cx: mean={cxs.mean():.3f}, cy: mean={cys.mean():.3f})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Toạ độ tâm X (chuẩn hoá 0 - 1)", fontsize=11)
    ax.set_ylabel("Toạ độ tâm Y (chuẩn hoá 0 - 1, 0 ở đỉnh)", fontsize=11)

    fig.tight_layout()
    fname = out_dir / "04_box_center_heatmap.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    print(f"  cx: mean={cxs.mean():.3f}, cy={cys.mean():.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# 05. Images per Split (chỉ dành cho Processed)
# ──────────────────────────────────────────────────────────────────────────────

def plot_images_per_split(
    split_box_counts: dict[str, tuple[int, int]],
    out_dir: Path,
    tag: str = "",
) -> None:
    """Figure 05 (cho PROCESSED): Số lượng ảnh và bounding box theo từng split."""
    splits = [s for s in ("train", "val", "test") if s in split_box_counts]
    if not splits:
        splits = list(split_box_counts.keys())

    n_imgs = [split_box_counts[s][0] for s in splits]
    n_boxes = [split_box_counts[s][1] for s in splits]
    x = np.arange(len(splits))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    title_suffix = f" – {tag.upper()}" if tag else ""
    fig.suptitle(f"05. Số lượng Ảnh và Bounding Box theo Split{title_suffix}", fontsize=14, fontweight="bold")

    colors_img = ["#1D3557", "#2A9D8F", "#7209B7"]
    colors_box = ["#457B9D", "#E76F51", "#9D4EDD"]

    bars1 = ax1.bar(x, n_imgs, color=colors_img[:len(splits)], width=0.6, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.upper() for s in splits], fontsize=11, fontweight="bold")
    ax1.set_ylabel("Số lượng ảnh", fontsize=11)
    ax1.set_title("1. Số lượng ảnh theo split", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    for bar, v in zip(bars1, n_imgs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(n_imgs) * 0.015,
                 f"{v:,}", ha="center", fontsize=10, fontweight="bold")

    bars2 = ax2.bar(x, n_boxes, color=colors_box[:len(splits)], width=0.6, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.upper() for s in splits], fontsize=11, fontweight="bold")
    ax2.set_ylabel("Số lượng Bounding Box", fontsize=11)
    ax2.set_title("2. Số lượng Bounding Box theo split", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    for bar, v in zip(bars2, n_boxes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(n_boxes) * 0.015,
                 f"{v:,}", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fname = out_dir / "05_images_per_split.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    for s in splits:
        ni, nb = split_box_counts[s]
        print(f"  Split {s}: {ni:,} ảnh, {nb:,} box, trung bình {nb/max(ni,1):.2f} box/ảnh")


# ──────────────────────────────────────────────────────────────────────────────
# 06. Tiny Box Candidates
# ──────────────────────────────────────────────────────────────────────────────

def plot_tiny_box_candidates(
    boxes: list[dict[str, Any]],
    out_dir: Path,
    min_box_px: int = 8,
    image_size: int = 640,
    tag: str = "",
) -> None:
    """Figure 06: Kiểm tra tỉ lệ bounding box siêu nhỏ (< min_box_px)."""
    canvas_h = int(round(image_size * 9 / 16)) if tag == "raw" else image_size
    ws = np.array([b["w"] * image_size for b in boxes], dtype=float)
    hs = np.array([b["h"] * canvas_h for b in boxes], dtype=float)
    min_dim = np.minimum(ws, hs)

    tiny_mask = min_dim < min_box_px
    n_tiny = int(tiny_mask.sum())
    pct = (n_tiny / len(boxes) * 100) if boxes else 0.0

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5.5))
    title_suffix = f" – {tag.upper()}" if tag else ""
    fig.suptitle(
        f"06. Khảo sát Bounding Box Siêu nhỏ (< {min_box_px}px){title_suffix}\n"
        f"(Tổng box: {len(boxes):,} | Box nhỏ: {n_tiny:,} | Tỉ lệ: {pct:.2f}%)",
        fontsize=13,
        fontweight="bold",
    )

    # Histogram
    ax0.hist(min_dim[min_dim <= 40], bins=40, range=(0, 40), color="#457B9D", edgecolor="white", alpha=0.9)
    ax0.axvline(min_box_px, color="#E63946", linestyle="--", linewidth=2, label=f"Ngưỡng lọc = {min_box_px}px")
    ax0.set_xlabel("Kích thước cạnh nhỏ nhất min(w, h) (pixels)")
    ax0.set_ylabel("Số lượng Box")
    ax0.set_title("1. Phân bố Cạnh nhỏ nhất (phạm vi ≤ 40px)", fontsize=11.5, fontweight="bold")
    ax0.legend()
    ax0.grid(axis="y", linestyle=":", alpha=0.5)

    # Scatter
    sample_size = min(len(boxes), 1500)
    idx = np.random.choice(len(boxes), sample_size, replace=False) if len(boxes) > 0 else []
    if len(idx):
        ax1.scatter(ws[idx], hs[idx], alpha=0.3, s=8, color="#2A9D8F", label="Box bình thường")
        tiny_idx = [i for i in idx if tiny_mask[i]]
        if tiny_idx:
            ax1.scatter(ws[tiny_idx], hs[tiny_idx], alpha=0.9, s=20, color="#E63946", label="Tiny Box")

    ax1.axhline(min_box_px, color="#E63946", linestyle="--", linewidth=1.5)
    ax1.axvline(min_box_px, color="#E63946", linestyle="--", linewidth=1.5, label=f"Ngưỡng {min_box_px}px")
    ax1.set_xlabel("Width (pixels)")
    ax1.set_ylabel("Height (pixels)")
    ax1.set_title("2. Tương quan Width vs Height", fontsize=11.5, fontweight="bold")
    ax1.legend()
    ax1.grid(linestyle=":", alpha=0.5)

    fig.tight_layout()
    fname = out_dir / "06_tiny_box_candidates.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    print(f"  Tiny box: {n_tiny}/{len(boxes):,} ({pct:.2f}%) có cạnh nhỏ hơn {min_box_px}px")


# ──────────────────────────────────────────────────────────────────────────────
# 08 (hoặc 07 ở raw): Brightness & Sharpness
# ──────────────────────────────────────────────────────────────────────────────

def analyze_and_plot_brightness_sharpness(
    img_paths: list[Path],
    out_dir: Path,
    max_samples: int = 1000,
    blur_threshold: float = 100.0,
    tag: str = "",
    fig_num: int = 8,
) -> dict[str, Any]:
    """Phân tích và vẽ phân bố độ sáng & độ sắc nét quang học."""
    if not img_paths:
        return {}

    cache_file = out_dir / "image_quality_cache.json"
    cache: dict[str, dict[str, Any]] = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    rng = random.Random(42)
    sample_paths = list(img_paths)
    if len(sample_paths) > max_samples:
        sample_paths = rng.sample(sample_paths, max_samples)

    brights = []
    sharps = []
    for p in sample_paths:
        stem = p.stem
        if stem in cache and "brightness" in cache[stem] and "sharpness" in cache[stem]:
            b = cache[stem]["brightness"]
            s = cache[stem]["sharpness"]
        else:
            res = calculate_brightness_sharpness(p)
            if res is None:
                continue
            b, s = res
            cache[stem] = {
                "brightness": round(b, 2),
                "sharpness": round(s, 2),
                "dark": b < 85,
                "bright": b > 185,
                "blur": s < blur_threshold,
            }
        brights.append(b)
        sharps.append(s)

    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    if not brights:
        return {}

    b_arr = np.array(brights)
    s_arr = np.array(sharps)
    blur_pct = float((s_arr < blur_threshold).sum() / len(s_arr) * 100)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    title_suffix = f" – {tag.upper()}" if tag else ""
    fig.suptitle(f"{fig_num:02d}. Đánh giá Quang học: Độ sáng & Độ sắc nét{title_suffix}", fontsize=14, fontweight="bold")

    # Brightness
    axes[0, 0].hist(b_arr, bins=40, range=(0, 255), color="#E9C46A", edgecolor="white", alpha=0.9)
    axes[0, 0].axvline(b_arr.mean(), color="#1D3557", linestyle="--", linewidth=2, label=f"Mean = {b_arr.mean():.1f}")
    axes[0, 0].set_title("1. Phân bố Độ sáng (Grayscale Mean [0, 255])", fontsize=11.5, fontweight="bold")
    axes[0, 0].set_xlabel("Độ sáng")
    axes[0, 0].set_ylabel("Số lượng ảnh")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", linestyle=":", alpha=0.5)

    # Sharpness
    s_plot = np.clip(s_arr, 0, 2000)
    axes[0, 1].hist(s_plot, bins=40, color="#2A9D8F", edgecolor="white", alpha=0.9)
    axes[0, 1].axvline(blur_threshold, color="#E63946", linestyle="--", linewidth=2, label=f"Ngưỡng mờ = {blur_threshold:.0f}")
    axes[0, 1].set_title(f"2. Phân bố Độ sắc nét (Laplacian Var | Ảnh mờ: {blur_pct:.1f}%)", fontsize=11.5, fontweight="bold")
    axes[0, 1].set_xlabel("Độ sắc nét")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.5)

    # Scatter
    scatter = axes[1, 0].scatter(b_arr, s_plot, c=b_arr, cmap="plasma", alpha=0.5, s=16)
    axes[1, 0].axhline(blur_threshold, color="#E63946", linestyle="--", label="Ngưỡng mờ")
    axes[1, 0].set_title("3. Tương quan Độ sáng vs Độ sắc nét", fontsize=11.5, fontweight="bold")
    axes[1, 0].set_xlabel("Độ sáng")
    axes[1, 0].set_ylabel("Độ nét")
    axes[1, 0].legend()
    axes[1, 0].grid(linestyle=":", alpha=0.5)
    plt.colorbar(scatter, ax=axes[1, 0], label="Độ sáng")

    # Donut Chart
    ax_d = axes[1, 1]
    clear_cnt = int(((s_arr >= blur_threshold) & (b_arr >= 60) & (b_arr <= 190)).sum())
    blur_cnt = int((s_arr < blur_threshold).sum())
    dark_cnt = int(((s_arr >= blur_threshold) & (b_arr < 60)).sum())
    bright_cnt = int(((s_arr >= blur_threshold) & (b_arr > 190)).sum())

    labels_q = ["Rõ nét & Chuẩn sáng", "Bị mờ (Blur)", "Thiếu sáng (Tối)", "Chói sáng"]
    counts_q = [clear_cnt, blur_cnt, dark_cnt, bright_cnt]
    colors_q = ["#2A9D8F", "#E63946", "#457B9D", "#F4A261"]

    valid_q = [(l, c, col) for l, c, col in zip(labels_q, counts_q, colors_q) if c > 0]
    if valid_q:
        vl, vc, vcol = zip(*valid_q)
        tot_q = sum(vc)
        wedges, texts, autotexts = ax_d.pie(
            vc, labels=None, colors=vcol,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 5.0 else "",
            startangle=140, pctdistance=0.72,
            textprops={"fontsize": 11, "fontweight": "bold", "color": "white"},
            wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.42},
        )
        ax_d.text(0, 0.08, f"{clear_cnt/tot_q*100:.1f}%", ha="center", va="center", fontsize=16, fontweight="bold", color="#1D3557")
        ax_d.text(0, -0.14, "Đạt chuẩn", ha="center", va="center", fontsize=10.5, color="#6C757D")
        leg_labels = [f"{l}: {c:,} ({c/tot_q*100:.1f}%)" for l, c in zip(vl, vc)]
        ax_d.legend(wedges, leg_labels, title=f"Phân loại ({tot_q:,} ảnh)", loc="center left", bbox_to_anchor=(0.95, 0.5), fontsize=9.5)
    ax_d.set_title("4. Tỉ lệ Phân loại Chất lượng Quang học", fontsize=11.5, fontweight="bold")

    fig.tight_layout()
    fname = out_dir / f"{fig_num:02d}_brightness_sharpness.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    print(f"  Độ sáng trung bình: {b_arr.mean():.1f}, Độ nét trung vị: {np.median(s_arr):.1f}, Ảnh mờ: {blur_pct:.2f}%")

    return {
        "samples_analyzed": len(b_arr),
        "mean_brightness": round(float(b_arr.mean()), 2),
        "median_sharpness": round(float(np.median(s_arr)), 2),
        "blur_pct": round(blur_pct, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 09 (hoặc 08 ở raw): Demo Ảnh Mẫu
# ──────────────────────────────────────────────────────────────────────────────

def plot_sample_labeled_images(
    data_root: Path,
    boxes: list[dict[str, Any]],
    classes: dict[int, str],
    out_dir: Path,
    num_samples: int = 9,
    seed: int = 42,
    tag: str = "",
    fig_num: int = 9,
) -> None:
    """Figure 09: Demo 9 ảnh mẫu kèm nhãn bounding box và tên tiếng Việt."""
    img_dir = data_root / "images"
    search_dir = img_dir if img_dir.is_dir() else data_root

    stem_to_boxes: dict[str, list[dict[str, Any]]] = {}
    for b in boxes:
        stem_to_boxes.setdefault(b["stem"], []).append(b)

    valid_stems = [st for st, bxs in stem_to_boxes.items() if bxs]
    if not valid_stems:
        return

    rng = random.Random(seed)
    chosen_stems = rng.sample(valid_stems, min(num_samples, len(valid_stems)))

    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    axes = axes.flatten()

    for idx, stem in enumerate(chosen_stems):
        ax = axes[idx]
        img_candidates = list(search_dir.rglob(f"{stem}.jpg")) + list(search_dir.rglob(f"{stem}.png"))
        if not img_candidates:
            ax.axis("off")
            continue

        img_p = img_candidates[0]
        try:
            im = Image.open(img_p).convert("RGB")
            w, h = im.size
            ax.imshow(im)

            bxs = stem_to_boxes.get(stem, [])
            for b in bxs:
                cx, cy, bw, bh = b["cx"] * w, b["cy"] * h, b["w"] * w, b["h"] * h
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                rect = patches.Rectangle((x1, y1), bw, bh, linewidth=2, edgecolor="#E63946", facecolor="none")
                ax.add_patch(rect)
                cid = b["class_id"]
                cname = classes.get(cid, f"Lớp {cid}")[:14]
                ax.text(x1, max(y1 - 4, 0), f"[{cid}] {cname}", color="white", fontsize=8.5, fontweight="bold",
                        bbox=dict(facecolor="#E63946", alpha=0.85, pad=1, edgecolor="none"))
        except Exception:
            pass

        ax.set_title(f"{img_p.name} ({len(bxs)} box)", fontsize=10, fontweight="bold")
        ax.axis("off")

    for j in range(len(chosen_stems), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fname = out_dir / f"{fig_num:02d}_sample_labels.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Phân tích Độc lập cho Raw và Processed
# ──────────────────────────────────────────────────────────────────────────────

def analyze_processed_dataset(
    data_root: str | Path = "data/processed",
    figures_dir: str | Path | None = None,
    class_table: str | Path | None = None,
    image_size: int = 640,
    min_box_px: int = 8,
    num_samples: int = 9,
    quality_samples: int = 1000,
    dhash_size: int = 32,
    dhash_hamming: int = 370,
    seed: int = 42,
) -> dict[str, Any]:
    """Hàm phân tích chuyên sâu độc lập cho bộ dữ liệu sau tiền xử lý (data/processed).

    Sinh đủ 9 biểu đồ chuẩn lưu tại analysis/figures/processed/:
      01_class_distribution: Phân bố 52 lớp trên 3 subplot (Train Real vs Aug, Val, Test)
      02_boxes_per_image: Phân bố box/ảnh sau letterbox
      03_box_dimensions: Kích thước box trên canvas 640x640
      04_box_center_heatmap: Mật độ vị trí tâm box trên canvas 640x640
      05_images_per_split: Số lượng ảnh và box theo từng split
      06_tiny_box_candidates: Xác nhận 0% box nhỏ < min_box_px
      07_data_leakage: Ma trận trùng lặp dHash & khoảng cách Hamming (bỏ viền xám 114)
      08_brightness_sharpness: Đo kiểm độ sáng và độ nét sau tiền xử lý
      09_sample_labels: Demo 9 ảnh mẫu sau tiền xử lý kèm bounding box
    Lưu tóm tắt kết quả: analysis/figures/processed/eda_summary.json
    """
    repo_root = Path(__file__).resolve().parents[1]
    data_root = Path(data_root).resolve()
    target_dir = Path(figures_dir).resolve() if figures_dir else repo_root / "analysis" / "figures" / "processed"
    target_dir.mkdir(parents=True, exist_ok=True)

    c_table = Path(class_table).resolve() if class_table else repo_root / "configs" / "classes.csv"
    classes = _load_classes(c_table)

    print(f"[EDA Processed] ========================================================")
    print(f"[EDA Processed] Phân tích dataset sau tiền xử lý tại: {data_root}")
    print(f"[EDA Processed] Thư mục xuất biểu đồ: {target_dir}")
    print(f"[EDA Processed] ========================================================")

    img_paths = _get_image_paths(data_root)
    stem_to_path = {p.stem: p for p in img_paths}

    split_names = ["train", "val", "test"]
    split_stems: dict[str, list[str]] = {}
    split_stats: dict[str, tuple[int, int]] = {}
    all_boxes: list[dict[str, Any]] = []

    for sname in split_names:
        s_lbl_dir = data_root / "labels" / sname
        s_img_dir = data_root / "images" / sname
        s_imgs = list(s_img_dir.glob("*.jpg")) if s_img_dir.is_dir() else []
        sboxes = _read_yolo_labels(s_lbl_dir, image_size_px=image_size)
        for b in sboxes:
            b["split"] = sname
            b["is_augmented"] = ("_aug" in b.get("stem", ""))
        all_boxes.extend(sboxes)
        stems = [p.stem for p in s_imgs] if s_imgs else list({b["stem"] for b in sboxes})
        split_stems[sname] = sorted(stems)
        split_stats[sname] = (len(stems), len(sboxes))

    summary_meta: dict[str, Any] = {
        "tag": "processed",
        "data_root": str(data_root),
        "n_images": len(img_paths),
        "n_boxes": len(all_boxes),
        "splits": split_stats,
    }

    # 01. Class distribution (3 subplot)
    print("\n[EDA Processed] 01 Class distribution (3 subplot: Train Real vs Aug, Val, Test)...")
    summary_meta["classes"] = plot_class_distribution(all_boxes, classes, target_dir / "01_class_distribution.png", tag="processed")

    # 02. Boxes per Image
    print("\n[EDA Processed] 02 Boxes per label / image distribution...")
    plot_boxes_per_image(data_root, all_boxes, target_dir, tag="processed")

    # 03. Box dimensions
    print("\n[EDA Processed] 03 Box dimensions...")
    plot_box_dimensions(all_boxes, target_dir, image_size=image_size, tag="processed")

    # 04. Box center heatmap
    print("\n[EDA Processed] 04 Box center heatmap...")
    plot_box_center_heatmap(all_boxes, target_dir, tag="processed")

    # 05. Images per split
    print("\n[EDA Processed] 05 Images per split...")
    plot_images_per_split(split_stats, target_dir, tag="processed")

    # 06. Tiny box candidates
    print("\n[EDA Processed] 06 Tiny box candidates...")
    plot_tiny_box_candidates(all_boxes, target_dir, min_box_px=min_box_px, image_size=image_size, tag="processed")

    # 07. Data leakage check (loại bỏ padding xám 114)
    print(f"\n[EDA Processed] 07 Data leakage check (dHash size={dhash_size}, Hamming <= {dhash_hamming}, bỏ viền xám)...")
    leak_data = check_data_leakage(split_stems, stem_to_path, hash_size=dhash_size, hamming_thresh=dhash_hamming)
    plot_data_leakage(leak_data, target_dir / "07_data_leakage.png", tag="processed")
    summary_meta["leakage"] = {k: v for k, v in leak_data.items() if k not in ("min_dists",)}

    # 08. Brightness & Sharpness
    print(f"\n[EDA Processed] 08 Brightness & Sharpness (mẫu {min(quality_samples, len(img_paths))} ảnh)...")
    summary_meta["quality"] = analyze_and_plot_brightness_sharpness(img_paths, target_dir, max_samples=quality_samples, tag="processed", fig_num=8)

    # 09. Demo n ảnh mẫu
    print(f"\n[EDA Processed] 09 Demo {num_samples} ảnh mẫu kèm label...")
    plot_sample_labeled_images(data_root, all_boxes, classes, target_dir, num_samples=num_samples, seed=seed, tag="processed", fig_num=9)

    summary_path = target_dir / "eda_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_meta, f, indent=2, ensure_ascii=False)
    print(f"\n[EDA Processed] Đã lưu tóm tắt thống kê: {summary_path}")
    print(f"[EDA Processed] Hoàn thành phân tích! Toàn bộ 9 biểu đồ đã ghi tại: {target_dir}")

    return summary_meta


def analyze_raw_dataset(
    data_root: str | Path = "data/raw",
    figures_dir: str | Path | None = None,
    class_table: str | Path | None = None,
    image_size: int = 640,
    min_box_px: int = 8,
    num_samples: int = 9,
    quality_samples: int = 1000,
    dhash_size: int = 32,
    dhash_hamming: int = 370,
    phash_hamming: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Phân tích dữ liệu thô ban đầu (data/raw) với chuẩn 8 biểu đồ."""
    repo_root = Path(__file__).resolve().parents[1]
    data_root = Path(data_root).resolve()
    target_dir = Path(figures_dir).resolve() if figures_dir else repo_root / "analysis" / "figures" / "raw"
    target_dir.mkdir(parents=True, exist_ok=True)

    c_table = Path(class_table).resolve() if class_table else repo_root / "configs" / "classes.csv"
    classes = _load_classes(c_table)

    print(f"[EDA Raw] ========================================================")
    print(f"[EDA Raw] Phân tích dataset thô tại: {data_root}")
    print(f"[EDA Raw] Thư mục xuất biểu đồ: {target_dir}")
    print(f"[EDA Raw] ========================================================")

    img_paths = _get_image_paths(data_root)
    flat_lbl_dir = data_root / "labels" if (data_root / "labels").is_dir() else data_root
    all_boxes = _read_yolo_labels(flat_lbl_dir, image_size_px=image_size)
    stems_all = sorted(list({b["stem"] for b in all_boxes}))

    summary_meta: dict[str, Any] = {
        "tag": "raw",
        "data_root": str(data_root),
        "n_images": len(img_paths),
        "n_boxes": len(all_boxes),
        "splits": {"all": (len(img_paths) if img_paths else len(stems_all), len(all_boxes))},
    }

    # 01. Class distribution
    print("\n[EDA Raw] 01 Class distribution (toàn bộ 52 lớp sắp xếp giảm dần)...")
    summary_meta["classes"] = plot_class_distribution(all_boxes, classes, target_dir / "01_class_distribution.png", tag="raw")

    # 02. Boxes per Image
    print("\n[EDA Raw] 02 Boxes per label / image distribution...")
    plot_boxes_per_image(data_root, all_boxes, target_dir, tag="raw")

    # 03. Box dimensions
    print("\n[EDA Raw] 03 Box dimensions...")
    plot_box_dimensions(all_boxes, target_dir, image_size=image_size, tag="raw")

    # 04. Box center heatmap
    print("\n[EDA Raw] 04 Box center heatmap...")
    plot_box_center_heatmap(all_boxes, target_dir, tag="raw")

    # 05. dHash + pHash Clusters
    ph_str = f" + pHash 64-bit (Hamming <= {phash_hamming})" if phash_hamming else ""
    print(f"\n[EDA Raw] 05 Gom cụm dHash 1024-bit (size={dhash_size}, Hamming <= {dhash_hamming}){ph_str}...")
    cluster_res = cluster_images_dhash(
        img_paths,
        hash_size=dhash_size,
        hamming_thresh=dhash_hamming,
        phash_thresh=phash_hamming,
    )
    plot_dhash_clusters(cluster_res, target_dir / "05_dhash_clusters.png", tag="raw")
    summary_meta["clustering"] = {k: v for k, v in cluster_res.items() if k != "clusters"}
    with open(target_dir / "dhash_clusters.json", "w", encoding="utf-8") as f:
        json.dump(cluster_res, f, ensure_ascii=False, indent=2)

    # 06. Tiny box candidates
    print("\n[EDA Raw] 06 Tiny box candidates...")
    plot_tiny_box_candidates(all_boxes, target_dir, min_box_px=min_box_px, image_size=image_size, tag="raw")

    # 07. Brightness & Sharpness
    print(f"\n[EDA Raw] 07 Brightness & Sharpness (mẫu {min(quality_samples, len(img_paths))} ảnh)...")
    summary_meta["quality"] = analyze_and_plot_brightness_sharpness(img_paths, target_dir, max_samples=quality_samples, tag="raw", fig_num=7)

    # 08. Demo n ảnh mẫu
    print(f"\n[EDA Raw] 08 Demo {num_samples} ảnh mẫu kèm label...")
    plot_sample_labeled_images(data_root, all_boxes, classes, target_dir, num_samples=num_samples, seed=seed, tag="raw", fig_num=8)

    summary_path = target_dir / "eda_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_meta, f, indent=2, ensure_ascii=False)
    print(f"\n[EDA Raw] Đã lưu tóm tắt thống kê: {summary_path}")
    print(f"[EDA Raw] Hoàn thành phân tích! Toàn bộ 8 biểu đồ đã ghi tại: {target_dir}")

    return summary_meta


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
    p.add_argument("--dhash-hamming", type=int, default=370, help="Ngưỡng khoảng cách Hamming dHash gom cụm (default: 370)")
    p.add_argument("--phash-hamming", type=int, default=10, help="Ngưỡng khoảng cách Hamming pHash gộp cụm (default: 10)")
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
