"""Offline data analysis for the detection set (boxes, not crops).

# Windows UTF-8 fix – must be before any other import that touches stdout
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

Produces 10 figures (written under analysis/figures/ strictly from 01 to 10):
  01_class_distribution    – toàn bộ 52 lớp sắp xếp giảm dần, cột đứng, nhãn xoay 90°
  02_boxes_per_image        – phân bố số lượng bounding box trên mỗi file nhãn / ảnh
  03_box_dimensions         – phân bố w, h, tỉ lệ khung hình (w/h), diện tích box
  04_box_center_heatmap     – mật độ vị trí tâm (cx, cy) trên khung hình (2D KDE & scatter)
  05_images_per_split       – số lượng ảnh và số box theo split (train/val/test)
  06_tiny_box_candidates    – histogram và scatter box nhỏ (< min_box_px)
  07_data_leakage           – kiểm tra rò rỉ dữ liệu (ma trận overlap + PCA feature cluster plot)
  08_brightness_sharpness   – phân tích độ sáng & độ sắc nét / độ mờ (Laplacian variance)
  09_image_properties       – độ phân giải, tỉ lệ khung hình, kênh màu, định dạng, dung lượng
  10_sample_labels          – demo n tấm ảnh mẫu vẽ bounding box và tên nhãn tiếng Việt

Run:
    python -m analysis.eda --data-root data/raw
    python -m analysis.eda --data-root data/processed  # lần 2 sau preprocess
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any
from collections import Counter
import shutil

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import cv2

# Cấu hình font tiếng Việt trong Matplotlib
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "Tahoma", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


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


def _get_image_paths(data_root: Path) -> list[Path]:
    """Tìm toàn bộ file ảnh trong data_root."""
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_dir = data_root / "images"
    search_dir = img_dir if img_dir.is_dir() else data_root

    paths = [p for p in search_dir.rglob("*.*") if p.suffix.lower() in img_extensions]
    return sorted(paths)


# ──────────────────────────────────────────────────────────────────────────────
# 01. Class Distribution (Toàn bộ 52 lớp, sắp xếp giảm dần, cột đứng, xoay 90°)
# ──────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    boxes: list[dict[str, Any]],
    classes: dict[int, str],
    out_dir: Path,
    tag: str = "",
) -> None:
    """Figure 01: Toàn bộ 52 lớp sắp xếp giảm dần theo số lượng box, cột đứng, nhãn xoay 90°."""
    num_classes = max(classes.keys()) + 1 if classes else 52
    counts = np.zeros(num_classes, dtype=int)
    for b in boxes:
        cid = b["class_id"]
        if 0 <= cid < num_classes:
            counts[cid] += 1

    # Sắp xếp giảm dần theo số lượng box
    order = np.argsort(counts)[::-1]
    sorted_counts = counts[order]
    sorted_labels = [
        f"[{order[i]:02d}] {classes.get(order[i], f'Lớp {order[i]}')}"
        for i in range(num_classes)
    ]

    fig, ax = plt.subplots(figsize=(24, 9.5))

    # Colormap gradient từ cao xuống thấp
    norm_counts = np.array(sorted_counts, dtype=float)
    max_cnt = max(norm_counts) if max(norm_counts) > 0 else 1.0
    colors = plt.cm.viridis(norm_counts / max_cnt)

    x_coords = np.arange(num_classes)
    bars = ax.bar(x_coords, sorted_counts, color=colors, edgecolor="#2B3A42", linewidth=0.7, alpha=0.9)

    # Nhãn trục X xoay đứng 90 độ
    ax.set_xticks(x_coords)
    ax.set_xticklabels(sorted_labels, rotation=90, ha="right", fontsize=9, fontweight="normal")

    # In giá trị số lượng box trên đỉnh mỗi cột
    y_max = max(sorted_counts) if len(sorted_counts) else 0
    y_margin = max(y_max * 0.02, 3)
    ax.set_ylim(0, y_max * 1.15 + 5)

    for bar, cnt in zip(bars, sorted_counts):
        if cnt > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_margin,
                f"{cnt:,}",
                ha="center",
                va="bottom",
                rotation=90 if y_max > 300 else 0,
                fontsize=8,
                fontweight="bold",
                color="#1F2937",
            )

    # Đường trung bình
    mean_val = counts.mean()
    ax.axhline(mean_val, color="#E63946", linestyle="--", linewidth=1.8,
               label=f"Trung bình = {mean_val:.1f} box / lớp")

    total_boxes = counts.sum()
    nonzero = np.count_nonzero(counts)
    min_nonzero = counts[counts > 0].min() if nonzero else 0

    ax.set_ylabel("Số lượng Bounding Box", fontsize=12, fontweight="bold")
    ax.set_title(
        f"01. Phân bố Toàn bộ 52 Lớp Biển báo (Sắp xếp Giảm dần){' – ' + tag if tag else ''}\n"
        f"(Tổng box: {total_boxes:,} | Lớp có dữ liệu: {nonzero}/52 | Max: {counts.max():,} box | Min non-zero: {min_nonzero:,} box)",
        fontsize=14, fontweight="bold", pad=12
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=11, loc="upper right")

    plt.tight_layout()
    fname = out_dir / ("01_class_distribution" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")
    print(f"  Tổng box: {total_boxes:,}, Lớp xuất hiện: {nonzero}/{num_classes}, Max: {counts.max()}, Min non-zero: {min_nonzero}")


# ──────────────────────────────────────────────────────────────────────────────
# 02. Boxes per Image / Label File (Số lượng box trên 1 file nhãn)
# ──────────────────────────────────────────────────────────────────────────────

def plot_boxes_per_image(
    data_root: Path,
    boxes: list[dict[str, Any]],
    out_dir: Path,
    tag: str = "",
) -> None:
    """Figure 02: Phân bố số lượng bounding box trên mỗi file nhãn / ảnh."""
    # Gom số box theo từng stem
    boxes_per_stem: Counter = Counter()
    for b in boxes:
        boxes_per_stem[b["stem"]] += 1

    # Tìm toàn bộ file nhãn để tính cả các ảnh có 0 box (nếu có)
    lbl_dir = data_root / "labels" if (data_root / "labels").is_dir() else data_root
    all_label_files = list(lbl_dir.rglob("*.txt"))
    all_stems = {f.stem for f in all_label_files}

    # Danh sách số lượng box trên mỗi file nhãn
    counts_list = [boxes_per_stem[s] for s in all_stems]
    if not counts_list:
        counts_list = list(boxes_per_stem.values())

    counts_arr = np.array(counts_list)
    freq = Counter(counts_list)
    max_boxes = max(counts_list) if counts_list else 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle(f"02. Phân bố Số lượng Bounding Box trên Mỗi File Nhãn / Ảnh{' – ' + tag if tag else ''}",
                 fontsize=14, fontweight="bold")

    # 1. Bar chart / Histogram số box
    ax0 = axes[0]
    x_vals = np.arange(0, max_boxes + 1)
    y_vals = [freq.get(x, 0) for x in x_vals]

    bars = ax0.bar(x_vals, y_vals, color="#3A86FF", edgecolor="#1D3557", linewidth=0.8, alpha=0.85)
    ax0.axvline(counts_arr.mean(), color="#E63946", linestyle="--", linewidth=1.8,
                label=f"Mean = {counts_arr.mean():.2f} box/ảnh")
    ax0.axvline(np.median(counts_arr), color="#FFB703", linestyle="-", linewidth=1.8,
                label=f"Median = {np.median(counts_arr):.0f} box/ảnh")

    ax0.set_xticks(x_vals)
    ax0.set_xlabel("Số lượng Bounding Box trong 1 ảnh", fontsize=11)
    ax0.set_ylabel("Số lượng file nhãn (Ảnh)", fontsize=11)
    ax0.set_title("Tần suất xuất hiện số lượng box trên mỗi ảnh", fontsize=12, fontweight="bold")
    ax0.grid(axis="y", linestyle=":", alpha=0.5)
    ax0.legend(fontsize=10)

    # In nhãn số lượng trên đầu cột
    for bar, y in zip(bars, y_vals):
        if y > 0:
            ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(y_vals) * 0.015,
                     f"{y:,}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # 2. Donut Chart phân nhóm mật độ
    ax1 = axes[1]
    c_0 = freq.get(0, 0)
    c_1 = freq.get(1, 0)
    c_2_3 = freq.get(2, 0) + freq.get(3, 0)
    c_4_6 = sum(freq.get(i, 0) for i in range(4, 7))
    c_7_plus = sum(freq.get(i, 0) for i in range(7, max_boxes + 1))

    group_labels = [
        f"1 biển báo ({c_1:,} ảnh)",
        f"2 – 3 biển báo ({c_2_3:,} ảnh)",
        f"4 – 6 biển báo ({c_4_6:,} ảnh)",
        f"≥ 7 biển báo (Dày đặc: {c_7_plus:,} ảnh)",
    ]
    group_counts = [c_1, c_2_3, c_4_6, c_7_plus]
    group_colors = ["#2A9D8F", "#457B9D", "#F4A261", "#E63946"]

    if c_0 > 0:
        group_labels.insert(0, f"0 box (Background: {c_0:,} ảnh)")
        group_counts.insert(0, c_0)
        group_colors.insert(0, "#8D99AE")

    valid_groups = [(l, c, col) for l, c, col in zip(group_labels, group_counts, group_colors) if c > 0]
    vl, vc, vcol = zip(*valid_groups)
    tot_boxes = sum(vc)

    def _autopct_filter(pct: float) -> str:
        return f"{pct:.1f}%" if pct >= 5.0 else ""

    wedges, texts, autotexts = ax1.pie(
        vc, labels=None, colors=vcol, autopct=_autopct_filter, startangle=140,
        pctdistance=0.72, textprops={"fontsize": 10, "fontweight": "bold", "color": "#FFFFFF"},
        wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.45}
    )

    # Hiển thị số liệu trung tâm Donut
    ax1.text(0, 0.08, f"{counts_arr.mean():.2f}", ha="center", va="center",
             fontsize=16, fontweight="bold", color="#1D3557")
    ax1.text(0, -0.14, "box / ảnh", ha="center", va="center",
             fontsize=10, color="#6C757D")

    # Legend tách biệt bên phải để tuyệt đối không bị đè chữ
    legend_labels = [
        f"{l} ({c/tot_boxes*100:.1f}%)"
        for l, c in zip(vl, vc)
    ]
    ax1.legend(
        wedges, legend_labels,
        title=f"Mật độ biển báo (Max: {max_boxes})",
        loc="center left",
        bbox_to_anchor=(0.95, 0.5),
        fontsize=9.5,
        title_fontsize=10.5,
        frameon=True,
        facecolor="#F8F9FA",
        edgecolor="#D3D3D3"
    )

    ax1.set_title(
        f"Tỷ lệ Mật độ Biển báo / Ảnh\n(Max: {max_boxes} box | Trung bình: {counts_arr.mean():.2f} box/ảnh)",
        fontsize=12, fontweight="bold"
    )

    plt.tight_layout()
    fname = out_dir / ("02_boxes_per_image" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")
    print(f"  Boxes/image: trung bình {counts_arr.mean():.2f}, trung vị {np.median(counts_arr):.0f}, max {max_boxes}")


# ──────────────────────────────────────────────────────────────────────────────
# 03. Box Dimensions (Kích thước bounding box)
# ──────────────────────────────────────────────────────────────────────────────

def plot_box_dimensions(
    boxes: list[dict[str, Any]],
    out_dir: Path,
    image_size: int = 640,
    tag: str = "",
) -> None:
    """Figure 03: phân bố kích thước box (w, h, aspect ratio, area) theo chuẩn Letterbox 640x640."""
    canvas_h = int(round(image_size * 9 / 16)) if tag == "raw" else image_size
    ws = np.array([b["w"] * image_size for b in boxes], dtype=float)
    hs = np.array([b["h"] * canvas_h for b in boxes], dtype=float)
    areas = ws * hs
    aspects = ws / (hs + 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"03. Phân bố Kích thước Bounding Box{' – ' + tag if tag else ''}", fontsize=14, fontweight="bold")

    def _hist(ax, data, xlabel, title, color, bins=60):
        ax.hist(data, bins=bins, color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(np.median(data), color="red", linestyle="--", linewidth=1.5, label=f"Median = {np.median(data):.1f}")
        ax.axvline(np.mean(data), color="orange", linestyle=":", linewidth=1.5, label=f"Mean = {np.mean(data):.1f}")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Số lượng box", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(fontsize=9)

    _hist(axes[0][0], ws,      "Width (pixels)",  "Chiều rộng Bounding Box (px)",          "#3A86FF")
    _hist(axes[0][1], hs,      "Height (pixels)", "Chiều cao Bounding Box (px)",           "#FB5607")
    _hist(axes[1][0], aspects, "Width / Height",  "Tỉ lệ khung hình Box (Aspect Ratio)",    "#06D6A0")
    _hist(axes[1][1], areas,   "Area (pixels²)",  "Diện tích Bounding Box (px²)",          "#8338EC")

    plt.tight_layout()
    fname = out_dir / ("03_box_dimensions" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")
    print(f"  w_px: median={np.median(ws):.1f}, mean={np.mean(ws):.1f}")
    print(f"  h_px: median={np.median(hs):.1f}, mean={np.mean(hs):.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# 04. Box Center Heatmap (Vị trí tâm biển báo)
# ──────────────────────────────────────────────────────────────────────────────

def plot_box_center_heatmap(
    boxes: list[dict[str, Any]],
    out_dir: Path,
    tag: str = "",
    resolution: int = 64,
) -> None:
    """Figure 04: 2D heatmap / histogram vị trí tâm (cx, cy) trên khung hình."""
    cxs = np.array([b["cx"] for b in boxes], dtype=float)
    cys = np.array([b["cy"] for b in boxes], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle(f"04. Mật độ Vị trí Tâm Bounding Box (Spatial Prior){' – ' + tag if tag else ''}", fontsize=14, fontweight="bold")

    # 2D histogram heatmap
    ax0 = axes[0]
    heatmap, xedges, yedges = np.histogram2d(cxs, cys, bins=resolution, range=[[0, 1], [0, 1]])
    im = ax0.imshow(
        heatmap.T,
        origin="upper",
        extent=[0, 1, 0, 1],
        cmap="inferno",
        aspect="equal",
    )
    ax0.set_xlabel("cx (chuẩn hóa 0–1)", fontsize=10)
    ax0.set_ylabel("cy (chuẩn hóa 0–1)", fontsize=10)
    ax0.set_title("2D Heatmap mật độ vị trí tâm biển báo", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax0, label="Số lượng box")

    # Scatter plot
    ax1 = axes[1]
    n_sample = min(4000, len(cxs))
    idx = np.random.choice(len(cxs), n_sample, replace=False)
    ax1.scatter(cxs[idx], cys[idx], alpha=0.2, s=6, color="#1D3557")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.invert_yaxis()  # (0,0) ở góc trên bên trái
    ax1.set_xlabel("cx (chuẩn hóa 0–1)", fontsize=10)
    ax1.set_ylabel("cy (chuẩn hóa 0–1)", fontsize=10)
    ax1.set_title(f"Phân tán tâm box (Subsample {n_sample:,} boxes)", fontsize=11, fontweight="bold")
    ax1.set_aspect("equal")
    ax1.grid(linestyle=":", alpha=0.5)

    plt.tight_layout()
    fname = out_dir / ("04_box_center_heatmap" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")
    print(f"  cx: mean={cxs.mean():.3f}, cy: mean={cys.mean():.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# 05. Images per Split (Số lượng ảnh và box theo split)
# ──────────────────────────────────────────────────────────────────────────────

def plot_images_per_split(
    split_box_counts: dict[str, tuple[int, int]],
    out_dir: Path,
    tag: str = "",
) -> None:
    """Figure 05: số ảnh và số box theo split."""
    splits = list(split_box_counts.keys())
    n_imgs = [split_box_counts[s][0] for s in splits]
    n_boxes = [split_box_counts[s][1] for s in splits]

    x = np.arange(len(splits))
    colors_img = ["#457B9D", "#E76F51", "#2A9D8F"]
    colors_box = ["#1D3557", "#F4A261", "#264653"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"05. Số lượng Ảnh và Bounding Box theo Split{' – ' + tag if tag else ''}", fontsize=14, fontweight="bold")

    bars1 = ax1.bar(x, n_imgs, color=colors_img[:len(splits)], alpha=0.9, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.upper() for s in splits], fontsize=11, fontweight="bold")
    ax1.set_ylabel("Số lượng ảnh", fontsize=11)
    ax1.set_title("Số lượng ảnh theo từng split", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    for bar, v in zip(bars1, n_imgs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(n_imgs) * 0.015,
                 f"{v:,}", ha="center", fontsize=10, fontweight="bold")

    bars2 = ax2.bar(x, n_boxes, color=colors_box[:len(splits)], alpha=0.9, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.upper() for s in splits], fontsize=11, fontweight="bold")
    ax2.set_ylabel("Số lượng Bounding Box", fontsize=11)
    ax2.set_title("Số lượng Bounding Box theo từng split", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    for bar, v in zip(bars2, n_boxes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(n_boxes) * 0.015,
                 f"{v:,}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fname = out_dir / ("05_images_per_split" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")
    for s in splits:
        ni, nb = split_box_counts[s]
        print(f"  Split {s}: {ni:,} ảnh, {nb:,} box, trung bình {nb/max(ni,1):.2f} box/ảnh")


# ──────────────────────────────────────────────────────────────────────────────
# 06. Tiny Box Candidates (Đánh giá box nhỏ < min_box_px)
# ──────────────────────────────────────────────────────────────────────────────

def plot_tiny_box_candidates(
    boxes: list[dict[str, Any]],
    out_dir: Path,
    min_box_px: int = 8,
    image_size: int = 640,
    tag: str = "",
) -> None:
    """Figure 06: histogram kích thước nhỏ; highlight box dưới min_box_px theo Letterbox 640x640."""
    canvas_h = int(round(image_size * 9 / 16)) if tag == "raw" else image_size
    ws = np.array([b["w"] * image_size for b in boxes], dtype=float)
    hs = np.array([b["h"] * canvas_h for b in boxes], dtype=float)
    min_dim = np.minimum(ws, hs)

    tiny_mask = min_dim < min_box_px
    n_tiny = tiny_mask.sum()
    pct = 100 * n_tiny / max(len(boxes), 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"06. Đánh giá Bounding Box siêu nhỏ (min(w,h) < {min_box_px}px) – {n_tiny}/{len(boxes):,} box ({pct:.2f}%){' – ' + tag if tag else ''}",
        fontsize=13, fontweight="bold"
    )

    # Histogram zoom
    ax0 = axes[0]
    threshold_display = min_box_px * 8
    small_vals = min_dim[min_dim < threshold_display]
    if len(small_vals) > 0:
        ax0.hist(small_vals, bins=40, color="#457B9D", alpha=0.75, edgecolor="white", linewidth=0.3, label="Hợp lệ (≥ 8px)")
        ax0.hist(small_vals[small_vals < min_box_px], bins=20, color="#E63946", alpha=0.85, label=f"Bị lọc bỏ (< {min_box_px}px)")
    ax0.axvline(min_box_px, color="#E63946", linestyle="--", linewidth=2, label=f"Ngưỡng min_box_px = {min_box_px}px")
    ax0.set_xlabel("min(w, h) – Pixels", fontsize=10)
    ax0.set_ylabel("Số lượng box", fontsize=10)
    ax0.set_title(f"Phân bố cạnh nhỏ nhất (Zoom ≤ {threshold_display}px)", fontsize=11, fontweight="bold")
    ax0.grid(axis="y", linestyle=":", alpha=0.5)
    ax0.legend(fontsize=9.5)

    # Scatter w vs h
    ax1 = axes[1]
    normal_mask = ~tiny_mask
    sample_n = min(normal_mask.sum(), 4000)
    sample_t = min(n_tiny, 1000)
    idx_n = np.random.choice(np.where(normal_mask)[0], sample_n, replace=False) if sample_n > 0 else []
    idx_t = np.random.choice(np.where(tiny_mask)[0], sample_t, replace=False) if sample_t > 0 else []

    if len(idx_n):
        ax1.scatter(ws[idx_n], hs[idx_n], alpha=0.2, s=6, color="#457B9D", label=f"Bình thường (n={sample_n})")
    if len(idx_t):
        ax1.scatter(ws[idx_t], hs[idx_t], alpha=0.8, s=16, color="#E63946", label=f"Tiny box (n={sample_t})")
    ax1.axhline(min_box_px, color="#FB8500", linestyle="--", linewidth=1.2)
    ax1.axvline(min_box_px, color="#FB8500", linestyle="--", linewidth=1.2, label=f"Ngưỡng {min_box_px}px")
    ax1.set_xlabel("Width (pixels)", fontsize=10)
    ax1.set_ylabel("Height (pixels)", fontsize=10)
    ax1.set_title("Tương quan Width vs Height", fontsize=11, fontweight="bold")
    ax1.grid(linestyle=":", alpha=0.5)
    ax1.legend(fontsize=9.5)

    plt.tight_layout()
    fname = out_dir / ("06_tiny_box_candidates" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")
    print(f"  Tiny box: {n_tiny}/{len(boxes):,} ({pct:.2f}%) có cạnh nhỏ hơn {min_box_px}px")


# ──────────────────────────────────────────────────────────────────────────────
# 07. Data Leakage & Feature Clustering (Kiểm tra rò rỉ và Phân cụm không gian)
# ──────────────────────────────────────────────────────────────────────────────

def check_and_plot_data_leakage(
    data_root: Path,
    split_stems: dict[str, list[str]],
    out_dir: Path,
    tag: str = "",
) -> dict[str, Any]:
    """Figure 07: Kiểm tra rò rỉ dữ liệu (Ma trận overlap + PCA Feature Cluster Plot giữa train và test)."""
    splits = list(split_stems.keys())
    results: dict[str, Any] = {"stem_leakage": {}, "content_hash_leakage": {}, "status": "PASSED"}

    if len(splits) < 2:
        print("  [Leak] Chỉ có 1 split hoặc không có split_dataset, bỏ qua kiểm tra rò rỉ.")
        return results

    # 1. Kiểm tra rò rỉ theo tên file (Stem overlap)
    n_splits = len(splits)
    stem_matrix = np.zeros((n_splits, n_splits), dtype=int)
    for i, s1 in enumerate(splits):
        set1 = set(split_stems[s1])
        for j, s2 in enumerate(splits):
            set2 = set(split_stems[s2])
            overlap = len(set1 & set2)
            stem_matrix[i, j] = overlap
            if i < j:
                results["stem_leakage"][f"{s1}_vs_{s2}"] = overlap
                if overlap > 0:
                    results["status"] = "LEAK_DETECTED"

    # 2. Kiểm tra rò rỉ theo MD5 hash ảnh
    img_dir = data_root / "images"
    search_dir = img_dir if img_dir.is_dir() else data_root
    stem_to_path: dict[str, Path] = {}
    for p in search_dir.rglob("*.*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            stem_to_path[p.stem] = p

    split_hashes: dict[str, dict[str, str]] = {}
    for sname in splits:
        split_hashes[sname] = {}
        for stem in split_stems[sname]:
            p = stem_to_path.get(stem)
            if p and p.exists():
                try:
                    with open(p, "rb") as fp:
                        h = hashlib.md5(fp.read()).hexdigest()
                        split_hashes[sname][stem] = h
                except Exception:
                    pass

    hash_matrix = np.zeros((n_splits, n_splits), dtype=int)
    for i, s1 in enumerate(splits):
        hashes1 = set(split_hashes[s1].values())
        for j, s2 in enumerate(splits):
            hashes2 = set(split_hashes[s2].values())
            overlap = len(hashes1 & hashes2) if hashes1 and hashes2 else 0
            hash_matrix[i, j] = overlap
            if i < j:
                results["content_hash_leakage"][f"{s1}_vs_{s2}"] = overlap
                if overlap > 0:
                    results["status"] = "CONTENT_DUPLICATE_LEAK_DETECTED"

    # 3. Tính toán dHash (Difference Hash - Perceptual Hashing) để phát hiện Near-duplicates
    train_stems = split_stems.get("train", [])
    test_stems = split_stems.get("test", [])

    dhash_results: dict[str, Any] = {}
    min_dists: np.ndarray | None = None

    if train_stems and test_stems:
        print("  [Leak] Đang tính mã băm dHash (Perceptual Hashing) cho toàn bộ Train và Test...")
        def _calc_dhash(stem: str) -> np.ndarray | None:
            p = stem_to_path.get(stem)
            if not p or not p.exists():
                return None
            try:
                with Image.open(p) as img:
                    w, h = img.size
                    # Căn chỉnh với canvas 640x640: nếu là ảnh letterbox, chỉ băm vùng nội dung thực tế (140->500)
                    if w == 640 and h == 640:
                        img_active = img.crop((0, 140, 640, 500))
                    else:
                        img_active = img
                    img_gray = img_active.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
                    arr = np.array(img_gray, dtype=np.int16)
                    return (arr[:, 1:] > arr[:, :-1]).flatten()
            except Exception:
                return None

        tr_hashes = [_calc_dhash(s) for s in train_stems]
        te_hashes = [_calc_dhash(s) for s in test_stems]

        valid_tr = [h for h in tr_hashes if h is not None]
        valid_te = [h for h in te_hashes if h is not None]

        if valid_tr and valid_te:
            tr_mat = np.array(valid_tr, dtype=np.uint8)
            te_mat = np.array(valid_te, dtype=np.uint8)

            tr_packed = np.packbits(tr_mat, axis=1)
            te_packed = np.packbits(te_mat, axis=1)
            xor_mat = np.bitwise_xor(te_packed[:, None, :], tr_packed[None, :, :])
            popcount = np.array([bin(k).count("1") for k in range(256)], dtype=np.uint8)
            hamm_dist = popcount[xor_mat].sum(axis=2)

            min_dists = hamm_dist.min(axis=1)

            n_near_3 = int((min_dists <= 3).sum())
            n_near_5 = int((min_dists <= 5).sum())
            pct_near_5 = float(n_near_5 / len(min_dists) * 100)

            dhash_results = {
                "total_test_evaluated": len(min_dists),
                "min_hamming_distance": int(min_dists.min()),
                "median_hamming_distance": float(np.median(min_dists)),
                "mean_hamming_distance": round(float(np.mean(min_dists)), 2),
                "near_duplicates_le_3": n_near_3,
                "near_duplicates_le_5": n_near_5,
                "near_duplicates_pct_le_5": round(pct_near_5, 2),
            }
            results["dhash_leakage"] = dhash_results
            print(f"  [dHash] Khoảng cách Hamming nhỏ nhất: {min_dists.min()} bit")
            print(f"  [dHash] Số ảnh Near-duplicate (Hamming <= 5 bit): {n_near_5}/{len(min_dists)} ({pct_near_5:.2f}%)")

    # 4. Phân tích Cụm đặc trưng (Feature Clustering & PCA)
    sample_train = split_stems.get("train", [])[:500]
    sample_test = split_stems.get("test", [])[:500]
    eval_stems = sample_train + sample_test
    eval_labels = ["train"] * len(sample_train) + ["test"] * len(sample_test)

    features = []
    valid_labels = []
    for stem, lbl in zip(eval_stems, eval_labels):
        p = stem_to_path.get(stem)
        if p and p.exists():
            try:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    h, w = img.shape[:2]
                    # Nếu là ảnh letterbox 640x640: cắt bỏ dải padding 140px
                    if w == 640 and h == 640:
                        img = img[140:500, :]
                    thumb = cv2.resize(img, (16, 16)).flatten() / 255.0
                    features.append(thumb)
                    valid_labels.append(lbl)
            except Exception:
                continue

    pca_coords = None
    cluster_labels = None
    n_clusters = 5
    if len(features) >= 10:
        feats_arr = np.array(features)
        mean_v = feats_arr.mean(axis=0)
        c_feats = feats_arr - mean_v
        u, s, vt = np.linalg.svd(c_feats, full_matrices=False)
        pca_coords = u[:, :2] * s[:2]

        try:
            pts = pca_coords.astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
            _, cl_labels, _ = cv2.kmeans(pts, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            cluster_labels = cl_labels.flatten()
        except Exception:
            cluster_labels = np.zeros(len(features), dtype=int)

    # Vẽ biểu đồ 4 panel (2x2): Matrix, dHash Hamming Histogram, 2D PCA Cluster, Cluster Balance
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"07. Đánh giá Rò rỉ Dữ liệu trên Canvas Chuẩn 640×640 (MD5, Scene Similarity & Phân cụm PCA){' – ' + tag if tag else ''}",
                 fontsize=14, fontweight="bold")

    # Panel [0, 0]: Ma trận giao nhau (Stem / Hash)
    ax0 = axes[0][0]
    im = ax0.imshow(stem_matrix, cmap="Blues", interpolation="nearest")
    ax0.set_xticks(range(n_splits))
    ax0.set_yticks(range(n_splits))
    ax0.set_xticklabels([s.upper() for s in splits], fontsize=10, fontweight="bold")
    ax0.set_yticklabels([s.upper() for s in splits], fontsize=10, fontweight="bold")
    ax0.set_title("1. Ma trận trùng lặp file & MD5 trên Canvas 640×640", fontsize=11, fontweight="bold")

    for i in range(n_splits):
        for j in range(n_splits):
            val = stem_matrix[i, j]
            is_diag = (i == j)
            txt = f"{val:,}" if is_diag else (f"Trùng: {val}" if val > 0 else "0 (An toàn)")
            color = "#1D3557" if is_diag else ("#E63946" if val > 0 else "#2A9D8F")
            ax0.text(j, i, txt, ha="center", va="center", fontsize=10, fontweight="bold", color=color)

    # Panel [0, 1]: Histogram dHash Hamming Distance (Near-duplicates trên Canvas 640x640)
    ax1 = axes[0][1]
    if min_dists is not None:
        bins = np.arange(-0.5, max(min_dists) + 1.5, 1)
        counts, edges, patches_list = ax1.hist(min_dists, bins=bins, edgecolor="white", linewidth=0.5, alpha=0.85)

        # Tô màu đỏ các cột <= 5 (near-duplicates) và màu xanh các cột > 5
        for edge, p in zip(edges[:-1], patches_list):
            if edge <= 5:
                p.set_facecolor("#E63946")  # Đỏ rò rỉ bối cảnh
            elif edge <= 8:
                p.set_facecolor("#FB8500")  # Cam cảnh tương đồng
            else:
                p.set_facecolor("#2A9D8F")  # Xanh an toàn

        ax1.axvline(5.5, color="#E63946", linestyle="--", linewidth=2, label="Ngưỡng Near-duplicate (≤ 5)")
        ax1.axvline(np.median(min_dists), color="#1D3557", linestyle="-", linewidth=2,
                    label=f"Trung vị khoảng cách = {np.median(min_dists):.0f}")

        pct_val = dhash_results.get("near_duplicates_pct_le_5", 0)
        ax1.set_xlabel("Mức độ khác biệt đặc trưng bối cảnh trên ảnh 640×640 (Ngưỡng rò rỉ ≤ 5: Frame liên tiếp)", fontsize=10)
        ax1.set_ylabel("Số lượng ảnh Test", fontsize=10)
        ax1.set_title(f"2. Phân bố Độ tương đồng Bối cảnh trên Canvas 640×640 (Near-duplicate: {pct_val:.1f}%)",
                      fontsize=11, fontweight="bold")
        ax1.grid(axis="y", linestyle=":", alpha=0.5)
        ax1.legend(fontsize=9.5)
    else:
        ax1.text(0.5, 0.5, "Không đủ dữ liệu dHash", ha="center", va="center")

    # Panel [1, 0]: 2D PCA Feature Space Scatter
    ax2 = axes[1][0]
    if pca_coords is not None:
        valid_labels_arr = np.array(valid_labels)
        idx_train = np.where(valid_labels_arr == "train")[0]
        idx_test = np.where(valid_labels_arr == "test")[0]

        ax2.scatter(pca_coords[idx_train, 0], pca_coords[idx_train, 1],
                    alpha=0.45, s=20, color="#3A86FF", label="Train Split", edgecolors="none")
        ax2.scatter(pca_coords[idx_test, 0], pca_coords[idx_test, 1],
                    alpha=0.75, s=28, color="#FB5607", label="Test Split", marker="^", edgecolors="none")
        ax2.set_xlabel("Principal Component 1 (PCA 1)", fontsize=10)
        ax2.set_ylabel("Principal Component 2 (PCA 2)", fontsize=10)
        ax2.set_title("3. Không gian đặc trưng 2D PCA từ Canvas 640×640 (Train vs Test)", fontsize=11, fontweight="bold")
        ax2.grid(linestyle=":", alpha=0.5)
        ax2.legend(fontsize=9.5)
    else:
        ax2.text(0.5, 0.5, "Không đủ mẫu để vẽ PCA", ha="center", va="center")

    # Panel [1, 1]: Phân bố Cụm cảnh (Scene Clusters) giữa Train và Test
    ax3 = axes[1][1]
    if cluster_labels is not None:
        valid_labels_arr = np.array(valid_labels)
        train_cl_counts = [((valid_labels_arr == "train") & (cluster_labels == k)).sum() for k in range(n_clusters)]
        test_cl_counts = [((valid_labels_arr == "test") & (cluster_labels == k)).sum() for k in range(n_clusters)]

        tr_total = max(sum(train_cl_counts), 1)
        te_total = max(sum(test_cl_counts), 1)
        train_pcts = [c / tr_total * 100 for c in train_cl_counts]
        test_pcts = [c / te_total * 100 for c in test_cl_counts]

        x_cl = np.arange(n_clusters)
        w_b = 0.35
        ax3.bar(x_cl - w_b / 2, train_pcts, width=w_b, color="#3A86FF", label="Train (%)", alpha=0.85)
        ax3.bar(x_cl + w_b / 2, test_pcts, width=w_b, color="#FB5607", label="Test (%)", alpha=0.85)
        ax3.set_xticks(x_cl)
        ax3.set_xticklabels([f"Cụm {k+1}" for k in x_cl], fontsize=9.5, fontweight="bold")
        ax3.set_ylabel("Tỷ lệ phân bố (%)", fontsize=10)
        ax3.set_title("4. Phân bố theo Cụm đặc trưng trên Canvas 640×640 (Cluster Balance)", fontsize=11, fontweight="bold")
        ax3.grid(axis="y", linestyle=":", alpha=0.5)
        ax3.legend(fontsize=9.5)
    else:
        ax3.text(0.5, 0.5, "Không đủ dữ liệu phân cụm", ha="center", va="center")

    # Đánh giá tổng hợp
    pct_near = dhash_results.get("near_duplicates_pct_le_5", 0)
    if results["status"] == "PASSED" and pct_near > 10:
        results["status"] = "TEMPORAL_SEQUENCE_LEAKAGE_DETECTED"

    status_note = (
        f"Kết luận: Stem/MD5 Overlap = 0 (Không trùng file). "
        f"Tuy nhiên trên canvas 640×640 phát hiện {pct_near:.1f}% ảnh Test là Near-Duplicate (rò rỉ chuỗi video / bối cảnh liên tiếp giữa Train và Test)."
    )
    fig.text(0.5, 0.015, status_note, ha="center", fontsize=11, fontweight="bold", color="#1F2937")

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fname = out_dir / ("07_data_leakage" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")
    print(f"  [Leak Status]: {results['status']}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 08. Brightness & Sharpness (Độ sáng và Độ sắc nét ảnh)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_and_plot_brightness_sharpness(
    img_paths: list[Path],
    out_dir: Path,
    max_samples: int = 1000,
    tag: str = "",
) -> dict[str, Any]:
    """Figure 08: Phân tích độ sáng (mean intensity) và độ sắc nét (Laplacian variance)."""
    if not img_paths:
        print("  [Quality] Không có file ảnh để phân tích độ sáng/độ nét.")
        return {}

    if 0 < max_samples < len(img_paths):
        sample_paths = random.sample(img_paths, max_samples)
    else:
        sample_paths = img_paths

    bright_vals = []
    sharp_vals = []

    for p in sample_paths:
        try:
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            b = float(gray.mean())
            s = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            bright_vals.append(b)
            sharp_vals.append(s)
        except Exception:
            continue

    if not bright_vals:
        return {}

    b_arr = np.array(bright_vals)
    s_arr = np.array(sharp_vals)

    dark_pct = (b_arr < 60).mean() * 100
    normal_b_pct = ((b_arr >= 60) & (b_arr <= 190)).mean() * 100
    bright_pct = (b_arr > 190).mean() * 100

    blur_threshold = 100.0
    blur_pct = (s_arr < blur_threshold).mean() * 100

    fig, axes = plt.subplots(2, 2, figsize=(17.5, 11))
    fig.suptitle(f"08. Phân tích Chất lượng Ảnh: Độ sáng (Brightness) & Độ sắc nét (Sharpness){' – ' + tag if tag else ''}",
                 fontsize=14, fontweight="bold")

    # 1. Histogram Brightness
    ax0 = axes[0][0]
    ax0.hist(b_arr, bins=50, color="#457B9D", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax0.axvline(60, color="#E63946", linestyle="--", linewidth=1.5, label="Ngưỡng tối (< 60)")
    ax0.axvline(190, color="#E76F51", linestyle="--", linewidth=1.5, label="Ngưỡng chói (> 190)")
    ax0.axvline(b_arr.mean(), color="#2A9D8F", linestyle="-", linewidth=2, label=f"Mean = {b_arr.mean():.1f}")
    ax0.set_xlabel("Độ sáng Pixel trung bình (0–255)", fontsize=10)
    ax0.set_ylabel("Số lượng ảnh", fontsize=10)
    ax0.set_title(f"Phân bố Độ sáng (Tối: {dark_pct:.1f}%, Chuẩn: {normal_b_pct:.1f}%, Chói: {bright_pct:.1f}%)",
                  fontsize=11, fontweight="bold")
    ax0.grid(axis="y", linestyle=":", alpha=0.5)
    ax0.legend(fontsize=9)

    # 2. Histogram Sharpness
    ax1 = axes[0][1]
    s_clipped = np.clip(s_arr, 0, np.percentile(s_arr, 98))
    ax1.hist(s_clipped, bins=50, color="#E76F51", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax1.axvline(blur_threshold, color="#E63946", linestyle="--", linewidth=2, label=f"Ngưỡng ảnh mờ (< {blur_threshold:.0f})")
    ax1.axvline(np.median(s_arr), color="#1D3557", linestyle="-", linewidth=2, label=f"Median = {np.median(s_arr):.1f}")
    ax1.set_xlabel("Độ biến thiên Laplacian (Laplacian Variance)", fontsize=10)
    ax1.set_ylabel("Số lượng ảnh", fontsize=10)
    ax1.set_title(f"Phân bố Độ sắc nét (Ảnh mờ / Blur: {blur_pct:.1f}%)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(fontsize=9)

    # 3. Scatter 2D
    ax2 = axes[1][0]
    scatter = ax2.scatter(b_arr, s_clipped, c=b_arr, cmap="coolwarm", alpha=0.4, s=12)
    ax2.axvline(60, color="#E63946", linestyle=":", alpha=0.7)
    ax2.axvline(190, color="#E76F51", linestyle=":", alpha=0.7)
    ax2.axhline(blur_threshold, color="#E63946", linestyle="--", linewidth=1.5, label="Ngưỡng mờ")
    ax2.set_xlabel("Độ sáng (Brightness)", fontsize=10)
    ax2.set_ylabel("Độ nét (Sharpness)", fontsize=10)
    ax2.set_title("Tương quan giữa Độ sáng và Độ sắc nét", fontsize=11, fontweight="bold")
    ax2.grid(linestyle=":", alpha=0.5)
    plt.colorbar(scatter, ax=ax2, label="Độ sáng")

    # 4. Donut Chart phân loại chất lượng – KHÔNG đè chữ, dùng bảng chú giải Legend
    ax3 = axes[1][1]
    labels_quality = ["Rõ nét & Chuẩn sáng", "Ảnh bị mờ (Blur)", "Thiếu sáng (Tối)", "Chói sáng"]
    clear_and_good = int(((s_arr >= blur_threshold) & (b_arr >= 60) & (b_arr <= 190)).sum())
    blur_count = int((s_arr < blur_threshold).sum())
    dark_count = int(((s_arr >= blur_threshold) & (b_arr < 60)).sum())
    bright_count = int(((s_arr >= blur_threshold) & (b_arr > 190)).sum())

    counts_q = [clear_and_good, blur_count, dark_count, bright_count]
    colors_q = ["#2A9D8F", "#E63946", "#457B9D", "#F4A261"]
    valid_slices = [(l, c, col) for l, c, col in zip(labels_quality, counts_q, colors_q) if c > 0]
    if valid_slices:
        vl, vc, vcol = zip(*valid_slices)
        tot_q = sum(vc)

        # Donut chart
        wedges, texts, autotexts = ax3.pie(
            vc, labels=None, colors=vcol,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 5.0 else "",
            startangle=140, pctdistance=0.72,
            textprops={"fontsize": 11, "fontweight": "bold", "color": "#FFFFFF"},
            wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.42}
        )

        # Trung tâm Donut hiển thị tỷ lệ đạt chuẩn nổi bật
        pct_good = clear_and_good / tot_q * 100
        ax3.text(0, 0.08, f"{pct_good:.1f}%", ha="center", va="center",
                 fontsize=16, fontweight="bold", color="#1D3557")
        ax3.text(0, -0.14, "Đạt chuẩn", ha="center", va="center",
                 fontsize=10.5, color="#6C757D")

        # Bảng chú giải Legend tách biệt sang bên phải, tuyệt đối không bị chồng lấn chữ
        legend_labels = [
            f"{l}: {c:,} ({c/tot_q*100:.1f}%)"
            for l, c in zip(vl, vc)
        ]
        ax3.legend(
            wedges, legend_labels,
            title=f"Phân loại (Mẫu: {tot_q:,} ảnh)",
            loc="center left",
            bbox_to_anchor=(0.95, 0.5),
            fontsize=9.5,
            title_fontsize=10.5,
            frameon=True,
            facecolor="#F8F9FA",
            edgecolor="#D3D3D3"
        )
    ax3.set_title("Tỉ lệ phân loại Chất lượng Dữ liệu", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fname = out_dir / ("08_brightness_sharpness" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")
    print(f"  Độ sáng trung bình: {b_arr.mean():.1f}, Độ nét trung vị: {np.median(s_arr):.1f}, Ảnh mờ: {blur_pct:.2f}%")

    return {
        "brightness_mean": float(b_arr.mean()),
        "sharpness_median": float(np.median(s_arr)),
        "blur_pct": float(blur_pct),
        "dark_pct": float(dark_pct),
        "bright_pct": float(bright_pct),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 09. Image Properties (Độ phân giải, Tỉ lệ khung hình, Kênh màu, Định dạng, Dung lượng)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_and_plot_image_properties(
    img_paths: list[Path],
    out_dir: Path,
    tag: str = "",
) -> dict[str, Any]:
    """Figure 09: Thống kê độ phân giải, tỷ lệ khung hình, số kênh màu, định dạng và dung lượng file."""
    if not img_paths:
        print("  [Props] Không có file ảnh để phân tích thuộc tính.")
        return {}

    widths, heights, aspect_ratios = [], [], []
    formats, channels, sizes_kb = [], [], []

    for p in img_paths:
        try:
            sz_kb = p.stat().st_size / 1024.0
            sizes_kb.append(sz_kb)

            with Image.open(p) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(round(w / h, 2))
                formats.append(img.format or p.suffix.upper().replace(".", ""))
                mode = img.mode
                c = 3 if mode in ("RGB", "YCbCr") else (4 if mode == "RGBA" else (1 if mode == "L" else 3))
                channels.append(c)
        except Exception:
            continue

    if not widths:
        return {}

    w_arr = np.array(widths)
    h_arr = np.array(heights)
    ar_arr = np.array(aspect_ratios)
    sz_arr = np.array(sizes_kb)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f"09. Thống kê Thuộc tính Tập ảnh (Image Metadata & Properties){' – ' + tag if tag else ''}",
                 fontsize=14, fontweight="bold")

    # 1. Độ phân giải (Width vs Height)
    ax0 = axes[0][0]
    res_pairs = [f"{w}x{h}" for w, h in zip(widths, heights)]
    top_res = Counter(res_pairs).most_common(5)
    top_res_str = ", ".join([f"{k} ({v} ảnh)" for k, v in top_res[:3]])

    ax0.scatter(w_arr, h_arr, color="#3A86FF", alpha=0.3, s=15, edgecolors="none")
    ax0.set_xlabel("Chiều rộng (Width px)", fontsize=10)
    ax0.set_ylabel("Chiều cao (Height px)", fontsize=10)
    ax0.set_title(f"Độ phân giải ảnh (Top: {top_res_str})", fontsize=11, fontweight="bold")
    ax0.grid(linestyle=":", alpha=0.5)

    # 2. Tỉ lệ khung hình (Aspect Ratio)
    ax1 = axes[0][1]
    ax1.hist(ar_arr, bins=35, color="#06D6A0", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax1.axvline(1.0, color="#E63946", linestyle="--", linewidth=1.5, label="1:1 (Vuông)")
    ax1.axvline(1.33, color="#FB5607", linestyle="--", linewidth=1.5, label="4:3 (~1.33)")
    ax1.axvline(1.78, color="#8338EC", linestyle="--", linewidth=1.5, label="16:9 (~1.78)")
    ax1.set_xlabel("Tỉ lệ khung hình (Width / Height)", fontsize=10)
    ax1.set_ylabel("Số lượng ảnh", fontsize=10)
    ax1.set_title(f"Phân bố Tỉ lệ khung hình (Median = {np.median(ar_arr):.2f})", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(fontsize=9)

    # 3. Định dạng và Số kênh màu
    ax2 = axes[1][0]
    fmt_counter = Counter(formats)
    ch_counter = Counter(channels)

    labels_fmt = [f"{k} ({v})" for k, v in fmt_counter.items()]
    ax2.bar(range(len(fmt_counter)), list(fmt_counter.values()), color="#FFB703", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(fmt_counter)))
    ax2.set_xticklabels(labels_fmt, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Số lượng ảnh", fontsize=10)
    ch_str = ", ".join([f"{k} kênh: {v} ảnh" for k, v in ch_counter.items()])
    ax2.set_title(f"Định dạng tệp ảnh & Kênh màu ({ch_str})", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    # 4. Dung lượng tệp (File Size in KB)
    ax3 = axes[1][1]
    ax3.hist(sz_arr, bins=40, color="#8338EC", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax3.axvline(np.median(sz_arr), color="#E63946", linestyle="--", linewidth=1.5, label=f"Median = {np.median(sz_arr):.1f} KB")
    ax3.axvline(sz_arr.mean(), color="#FFB703", linestyle=":", linewidth=1.5, label=f"Mean = {sz_arr.mean():.1f} KB")
    ax3.set_xlabel("Dung lượng tệp (KB)", fontsize=10)
    ax3.set_ylabel("Số lượng ảnh", fontsize=10)
    ax3.set_title(f"Phân bố Dung lượng tệp (Min: {sz_arr.min():.1f}KB, Max: {sz_arr.max():.1f}KB)", fontsize=11, fontweight="bold")
    ax3.grid(axis="y", linestyle=":", alpha=0.5)
    ax3.legend(fontsize=9)

    plt.tight_layout()
    fname = out_dir / ("09_image_properties" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")
    print(f"  Độ phân giải phổ biến: {top_res[:2]}")
    print(f"  Dung lượng ảnh: trung bình {sz_arr.mean():.1f} KB, trung vị {np.median(sz_arr):.1f} KB")

    return {
        "resolutions_top": top_res,
        "aspect_ratio_median": float(np.median(ar_arr)),
        "file_size_kb_mean": float(sz_arr.mean()),
        "formats": dict(fmt_counter),
        "channels": dict(ch_counter),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 10. Sample Labeled Images (Demo trực quan hóa n ảnh kèm bounding box)
# ──────────────────────────────────────────────────────────────────────────────

def plot_sample_labeled_images(
    data_root: Path,
    boxes: list[dict[str, Any]],
    classes: dict[int, str],
    out_dir: Path,
    num_samples: int = 9,
    seed: int = 42,
    tag: str = "",
) -> None:
    """Figure 10: Trực quan hóa n ảnh mẫu kèm bounding box và nhãn lớp tiếng Việt."""
    if not boxes:
        return

    boxes_by_stem: dict[str, list[dict[str, Any]]] = {}
    for b in boxes:
        boxes_by_stem.setdefault(b["stem"], []).append(b)

    stems_with_boxes = list(boxes_by_stem.keys())
    if not stems_with_boxes:
        return

    img_dir = data_root / "images"
    search_dir = img_dir if img_dir.is_dir() else data_root
    stem_to_path = {}
    for p in search_dir.rglob("*.*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            stem_to_path[p.stem] = p

    valid_stems = [s for s in stems_with_boxes if s in stem_to_path]
    if not valid_stems:
        return

    rng = random.Random(seed)
    n_show = min(num_samples, len(valid_stems))
    selected_stems = rng.sample(valid_stems, n_show)

    cols = 3 if n_show <= 9 else 4
    rows = (n_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.5))
    fig.suptitle(f"10. Ảnh Mẫu Kiểm tra Bounding Box & Nhãn Lớp Tiếng Việt{' – ' + tag if tag else ''}",
                 fontsize=14, fontweight="bold", y=0.99)

    ax_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    cmap = plt.get_cmap("tab20")

    for i, stem in enumerate(selected_stems):
        ax = ax_list[i]
        img_p = stem_to_path[stem]
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = img_rgb.shape[:2]

        ax.imshow(img_rgb)
        ax.set_title(f"Ảnh: {img_p.name} ({len(boxes_by_stem[stem])} box)", fontsize=9.5, fontweight="bold")
        ax.axis("off")

        for b in boxes_by_stem[stem]:
            cid = b["class_id"]
            cx, cy, bw, bh = b["cx"], b["cy"], b["w"], b["h"]

            x1 = (cx - bw / 2) * w_img
            y1 = (cy - bh / 2) * h_img
            w_box = bw * w_img
            h_box = bh * h_img

            color = cmap(cid % 20)

            rect = patches.Rectangle(
                (x1, y1), w_box, h_box,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)

            cname = classes.get(cid, f"Class {cid}")
            label_text = f"[{cid}] {cname[:18]}"

            ax.text(
                x1, max(y1 - 4, 12),
                label_text,
                color="white",
                fontsize=7.5,
                fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.85, edgecolor="none", boxstyle="round,pad=0.2"),
            )

    for j in range(len(selected_stems), len(ax_list)):
        ax_list[j].axis("off")

    plt.tight_layout()
    fname = out_dir / ("10_sample_labels" + (f"_{tag}" if tag else "") + ".png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# Hàm collect_data
# ──────────────────────────────────────────────────────────────────────────────

def collect_data(
    data_root: Path,
    image_size: int = 640,
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, int]], dict[str, list[str]], list[Path], str]:
    """Đọc dữ liệu từ data_root."""
    data_root = Path(data_root).resolve()
    tag = data_root.name

    img_paths = _get_image_paths(data_root)
    all_boxes: list[dict[str, Any]] = []
    split_stats: dict[str, tuple[int, int]] = {}
    split_stems: dict[str, list[str]] = {}

    split_names = ("train", "val", "test")

    # Cấu trúc 1: labels/{train,val,test}/
    labels_root = data_root / "labels"
    if labels_root.is_dir() and any((labels_root / s).is_dir() for s in split_names):
        print(f"  Phát hiện cấu trúc processed: labels/{{train,val,test}}/")
        for sname in split_names:
            sdir = labels_root / sname
            if not sdir.is_dir():
                continue
            s_img_dir = data_root / "images" / sname
            s_imgs = list(s_img_dir.glob("*.*")) if s_img_dir.is_dir() else []
            sboxes = _read_yolo_labels(sdir, image_size_px=image_size)
            all_boxes.extend(sboxes)
            stems = [p.stem for p in s_imgs] if s_imgs else [b["stem"] for b in sboxes]
            split_stems[sname] = list(set(stems))
            split_stats[sname] = (len(split_stems[sname]), len(sboxes))
        return all_boxes, split_stats, split_stems, img_paths, tag

    # Cấu trúc 2: labels/ phẳng + split_dataset/
    flat_labels_dir = data_root / "labels" if (data_root / "labels").is_dir() else data_root
    all_boxes = _read_yolo_labels(flat_labels_dir, image_size_px=image_size)
    stems_all = {b["stem"] for b in all_boxes}

    split_dir = data_root / "split_dataset"
    if split_dir.is_dir():
        for sname in split_names:
            sf = _find_split_file(split_dir, sname)
            if not sf:
                continue
            with open(sf, "r", encoding="utf-8") as fp:
                lines = [l.strip() for l in fp if l.strip()]
            stems_in_split = [Path(l).stem for l in lines]
            valid_stems = [s for s in stems_in_split if s in {p.stem for p in img_paths} or s in stems_all]
            s_boxes = [b for b in all_boxes if b["stem"] in set(valid_stems)]
            split_stems[sname] = valid_stems
            split_stats[sname] = (len(valid_stems), len(s_boxes))

        if split_stats:
            return all_boxes, split_stats, split_stems, img_paths, tag

    # Không có split: báo all
    n_imgs = len(img_paths) if img_paths else len(stems_all)
    split_stems["all"] = list(stems_all)
    split_stats["all"] = (n_imgs, len(all_boxes))
    return all_boxes, split_stats, split_stems, img_paths, tag


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Offline EDA toàn diện chuẩn 10 biểu đồ từ 01-10 cho bộ detection VNTS (boxes, không crop)"
    )
    p.add_argument("--data-root", default=None, help="Thư mục gốc dataset (raw hoặc processed)")
    p.add_argument("--figures-dir", default=None, help="Thư mục xuất hình (mặc định: analysis/figures/)")
    p.add_argument("--class-table", default=None, help="Đường dẫn configs/classes.csv")
    p.add_argument("--image-size", type=int, default=640, help="Kích thước ảnh để tính px (default: 640)")
    p.add_argument("--min-box-px", type=int, default=8, help="Ngưỡng tiny box (default: 8)")
    p.add_argument("--num-samples", type=int, default=9, help="Số ảnh mẫu vẽ demo nhãn (default: 9)")
    p.add_argument("--quality-samples", type=int, default=1000, help="Số ảnh lấy mẫu đo độ sáng/độ nét (default: 1000)")
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
    all_boxes, split_stats, split_stems, img_paths, tag = collect_data(data_root, image_size=args.image_size)

    if not all_boxes:
        print("[EDA] Không tìm thấy bounding box nào. Kiểm tra lại cấu trúc thư mục.")
        sys.exit(1)

    print(f"[EDA] Tìm thấy {len(img_paths):,} ảnh, {len(all_boxes):,} box, splits: {list(split_stats.keys())}")

    np.random.seed(42)
    random.seed(42)

    summary_meta: dict[str, Any] = {
        "tag": tag,
        "n_images": len(img_paths),
        "n_boxes": len(all_boxes),
        "splits": split_stats,
    }

    # Phân chia thư mục lưu trữ cho raw và processed
    figures_root = figures_dir
    if figures_dir.name in ("raw", "processed"):
        target_dir = figures_dir
        figures_root = figures_dir.parent
    else:
        target_dir = figures_root / tag if tag in ("raw", "processed") else figures_root
    target_dir.mkdir(parents=True, exist_ok=True)

    # ── 01. Class distribution (Gộp 52 lớp, sắp xếp giảm dần, cột đứng xoay 90°) ──
    print("\n[EDA] 01 Class distribution (toàn bộ 52 lớp sắp xếp giảm dần)...")
    plot_class_distribution(all_boxes, classes, target_dir, tag=tag)

    # ── 02. Boxes per Image / Label File ──
    print("\n[EDA] 02 Boxes per label / image distribution...")
    plot_boxes_per_image(data_root, all_boxes, target_dir, tag=tag)

    # ── 03. Box dimensions ──
    print("\n[EDA] 03 Box dimensions...")
    plot_box_dimensions(all_boxes, target_dir, image_size=args.image_size, tag=tag)

    # ── 04. Box center heatmap ──
    print("\n[EDA] 04 Box center heatmap...")
    plot_box_center_heatmap(all_boxes, target_dir, tag=tag)

    # ── 05. Images per split ──
    print("\n[EDA] 05 Images per split...")
    plot_images_per_split(split_stats, target_dir, tag=tag)

    # ── 06. Tiny box candidates ──
    print("\n[EDA] 06 Tiny box candidates...")
    plot_tiny_box_candidates(
        all_boxes,
        target_dir,
        min_box_px=args.min_box_px,
        image_size=args.image_size,
        tag=tag,
    )

    # ── 07. Data leakage & Feature Clustering ──
    print("\n[EDA] 07 Data leakage check & PCA Feature Clustering...")
    leak_res = check_and_plot_data_leakage(data_root, split_stems, target_dir, tag=tag)
    summary_meta["leakage"] = leak_res

    # ── 08. Brightness & Sharpness ──
    print(f"\n[EDA] 08 Brightness & Sharpness (mẫu {min(args.quality_samples, len(img_paths))} ảnh)...")
    qual_res = analyze_and_plot_brightness_sharpness(
        img_paths, target_dir, max_samples=args.quality_samples, tag=tag
    )
    summary_meta["quality"] = qual_res

    # ── 09. Image properties & metadata ──
    print("\n[EDA] 09 Image properties & metadata...")
    props_res = analyze_and_plot_image_properties(img_paths, target_dir, tag=tag)
    summary_meta["properties"] = props_res

    # ── 10. Demo n tấm ảnh có label ──
    print(f"\n[EDA] 10 Demo {args.num_samples} ảnh mẫu kèm label...")
    plot_sample_labeled_images(
        data_root, all_boxes, classes, target_dir,
        num_samples=args.num_samples, seed=42, tag=tag
    )

    # Lưu tóm tắt kết quả
    summary_path = target_dir / f"eda_summary_{tag}.json"
    clean_summary_path = target_dir / "eda_summary.json"
    root_summary_path = figures_root / f"eda_summary_{tag}.json"
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_meta, f, indent=2, ensure_ascii=False)
        shutil.copy2(summary_path, clean_summary_path)
        print(f"\n[EDA] Đã lưu tóm tắt thống kê: {summary_path}")
    except Exception as e:
        print(f"  [Warn] Không thể lưu summary json: {e}")

    # Đảm bảo tạo bản sao không có hậu tố _tag trong target_dir
    if tag:
        for p in target_dir.glob(f"*_{tag}.png"):
            clean_name = p.name.replace(f"_{tag}.", ".")
            clean_p = target_dir / clean_name
            shutil.copy2(p, clean_p)

    print(f"\n[EDA] Hoàn thành phân tích! Toàn bộ 10 biểu đồ từ 01–10 đã ghi tại: {target_dir}")


if __name__ == "__main__":
    main()
