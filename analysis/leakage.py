"""Module chuyên trách phân tích và vẽ biểu đồ rò rỉ dữ liệu (Data Leakage).

Đặc điểm quan trọng:
  - Tự động loại bỏ hoàn toàn viền xám (114) sinh ra khi letterbox-resize
    để không nhận nhầm vùng xám là tương đồng nội dung.
  - Sử dụng dHash chuẩn size=32 (1024-bit) và ngưỡng Hamming <= 370.
  - Loại bỏ các câu chữ kết luận dài dòng trong biểu đồ theo yêu cầu người dùng.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from analysis.image_utils import calculate_dhash, crop_active_region


def check_data_leakage(
    split_stems: dict[str, list[str]],
    stem_to_path: dict[str, Path],
    hash_size: int = 32,
    hamming_thresh: int = 370,
) -> dict[str, Any]:
    """Khảo sát rò rỉ dữ liệu dHash giữa tất cả các cặp split (train, val, test)."""
    splits = [s for s in ("train", "val", "test") if s in split_stems and split_stems[s]]
    n_splits = len(splits)
    if n_splits < 2:
        return {"status": "SKIPPED_NOT_ENOUGH_SPLITS"}

    # 1. Tính mã băm dHash (bỏ viền xám 114) cho từng split
    dhash_cache: dict[str, np.ndarray | None] = {}

    def _get_hash(stem: str) -> np.ndarray | None:
        if stem in dhash_cache:
            return dhash_cache[stem]
        p = stem_to_path.get(stem)
        if not p or not p.exists():
            dhash_cache[stem] = None
            return None
        h = calculate_dhash(p, hash_size=hash_size, strip_padding=True)
        dhash_cache[stem] = h
        return h

    split_valid_stems: dict[str, list[str]] = {}
    split_packed: dict[str, np.ndarray] = {}
    for s in splits:
        valid = [st for st in split_stems[s] if _get_hash(st) is not None]
        split_valid_stems[s] = valid
        if valid:
            mat = np.array([_get_hash(st) for st in valid], dtype=np.uint8)
            split_packed[s] = np.packbits(mat, axis=1)
        else:
            split_packed[s] = np.zeros((0, (hash_size * hash_size) // 8), dtype=np.uint8)

    # 2. Xây dựng ma trận khoảng cách Hamming
    popcount = np.array([bin(k).count("1") for k in range(256)], dtype=np.uint8)
    leak_matrix = np.zeros((n_splits, n_splits), dtype=int)
    leak_pair_matrix = np.zeros((n_splits, n_splits), dtype=int)
    all_leak_pairs: list[dict[str, Any]] = []
    leak_summary_by_split: dict[str, Any] = {}

    for i, s1 in enumerate(splits):
        for j, s2 in enumerate(splits):
            if i == j:
                leak_matrix[i, j] = len(split_stems[s1])
                leak_pair_matrix[i, j] = len(split_stems[s1])
                continue

            p1 = split_packed.get(s1)
            p2 = split_packed.get(s2)
            if p1 is None or p2 is None or len(p1) == 0 or len(p2) == 0:
                continue

            xor = np.bitwise_xor(p1[:, None, :], p2[None, :, :])
            dist = popcount[xor].sum(axis=2)

            leaked_i, leaked_j = np.where(dist <= hamming_thresh)
            n_pairs = len(leaked_i)
            n_unique_s1 = len(np.unique(leaked_i))
            n_unique_s2 = len(np.unique(leaked_j))

            leak_matrix[i, j] = n_unique_s1
            leak_pair_matrix[i, j] = n_pairs

            if i < j:
                split_pair_key = f"{s1}_vs_{s2}"
                leak_summary_by_split[split_pair_key] = {
                    "leak_pairs_count": int(n_pairs),
                    f"{s1}_images_leaked": int(n_unique_s1),
                    f"{s2}_images_leaked": int(n_unique_s2),
                    f"{s1}_leak_pct": round(float(n_unique_s1 / max(len(split_stems[s1]), 1) * 100), 2),
                    f"{s2}_leak_pct": round(float(n_unique_s2 / max(len(split_stems[s2]), 1) * 100), 2),
                }

                for k in range(n_pairs):
                    idx1 = leaked_i[k]
                    idx2 = leaked_j[k]
                    st1 = split_valid_stems[s1][idx1]
                    st2 = split_valid_stems[s2][idx2]
                    all_leak_pairs.append({
                        "split_a": s1,
                        "image_a": stem_to_path[st1].name if st1 in stem_to_path else f"{st1}.jpg",
                        "split_b": s2,
                        "image_b": stem_to_path[st2].name if st2 in stem_to_path else f"{st2}.jpg",
                        "hamming_distance": int(dist[idx1, idx2]),
                    })

    # 3. Đánh giá riêng chi tiết Test vs Train
    dhash_test_to_train: dict[str, Any] = {}
    min_dists: np.ndarray | None = None
    if "train" in split_packed and "test" in split_packed:
        p_tr = split_packed["train"]
        p_te = split_packed["test"]
        if len(p_tr) > 0 and len(p_te) > 0:
            xor_te_tr = np.bitwise_xor(p_te[:, None, :], p_tr[None, :, :])
            hamm_te_tr = popcount[xor_te_tr].sum(axis=2)
            min_dists = hamm_te_tr.min(axis=1)

            n_near_thresh = int(np.count_nonzero(min_dists <= hamming_thresh))
            pct_near_thresh = float(n_near_thresh / len(min_dists) * 100)
            dhash_test_to_train = {
                "min_hamming_distance": int(min_dists.min()),
                "median_hamming_distance": float(np.median(min_dists)),
                "near_duplicates_count": n_near_thresh,
                "near_duplicates_pct": round(pct_near_thresh, 2),
                "hamming_threshold": hamming_thresh,
            }

    total_leaks = len(all_leak_pairs)
    status = "PASSED" if total_leaks == 0 else "LEAK_DETECTED"

    return {
        "status": status,
        "hash_size": hash_size,
        "dhash_threshold": hamming_thresh,
        "splits": splits,
        "split_stems": split_stems,
        "total_leak_pairs": total_leaks,
        "leak_matrix": leak_matrix.tolist(),
        "leak_pair_matrix": leak_pair_matrix.tolist(),
        "leak_summary_by_split": leak_summary_by_split,
        "dhash_test_to_train": dhash_test_to_train,
        "leak_pairs": all_leak_pairs,
        "min_dists": min_dists,
    }


def find_leaked_test_stems(
    train_stems: list[str],
    test_stems: list[str],
    stem_to_path: dict[str, Path],
    hash_size: int = 32,
    hamming_thresh: int = 370,
) -> tuple[list[str], list[str]]:
    """Phát hiện và phân tách các stem trong test split thành clean và leaked."""
    tr_hashes = [calculate_dhash(stem_to_path[s], hash_size=hash_size, strip_padding=True) for s in train_stems if s in stem_to_path]
    te_hashes = [calculate_dhash(stem_to_path[s], hash_size=hash_size, strip_padding=True) for s in test_stems if s in stem_to_path]

    valid_tr = [h for h in tr_hashes if h is not None]
    valid_te_stems = [s for s in test_stems if s in stem_to_path and calculate_dhash(stem_to_path[s], hash_size=hash_size, strip_padding=True) is not None]
    valid_te = [h for h in te_hashes if h is not None]

    if not valid_tr or not valid_te:
        return list(test_stems), []

    tr_packed = np.packbits(np.array(valid_tr, dtype=np.uint8), axis=1)
    te_packed = np.packbits(np.array(valid_te, dtype=np.uint8), axis=1)
    popcount = np.array([bin(k).count("1") for k in range(256)], dtype=np.uint8)

    xor = np.bitwise_xor(te_packed[:, None, :], tr_packed[None, :, :])
    min_dists = popcount[xor].sum(axis=2).min(axis=1)

    clean_stems: list[str] = []
    leaked_stems: list[str] = []
    for stem, dist in zip(valid_te_stems, min_dists):
        if dist <= hamming_thresh:
            leaked_stems.append(stem)
        else:
            clean_stems.append(stem)

    return clean_stems, leaked_stems


def plot_data_leakage(
    leak_data: dict[str, Any],
    out_path: Path,
    tag: str = "",
) -> None:
    """Vẽ Figure 07: Ma trận dHash Overlap và Histogram khoảng cách Hamming.

    Thiết kế gọn gàng, trực quan, loại bỏ các đoạn văn bản kết luận dài dòng.
    """
    splits = leak_data.get("splits", ["train", "val", "test"])
    n_splits = len(splits)
    leak_pair_matrix = np.array(leak_data.get("leak_pair_matrix", np.zeros((n_splits, n_splits))))
    leak_matrix = np.array(leak_data.get("leak_matrix", np.zeros((n_splits, n_splits))))
    split_stems = leak_data.get("split_stems", {})
    min_dists = leak_data.get("min_dists")
    hash_size = leak_data.get("hash_size", 32)
    hamming_thresh = leak_data.get("dhash_threshold", 370)
    dhash_test_to_train = leak_data.get("dhash_test_to_train", {})

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    title_suffix = f" – {tag.upper()}" if tag else ""
    fig.suptitle(
        f"07. Đánh giá Rò rỉ Dữ liệu dHash (size={hash_size}, Hamming ≤ {hamming_thresh}){title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    # Panel 0: Ma trận trùng lặp
    ax0 = axes[0]
    rgba_grid = np.zeros((n_splits, n_splits, 4), dtype=float)
    for i in range(n_splits):
        for j in range(n_splits):
            if i == j:
                rgba_grid[i, j] = [0.89, 0.95, 0.99, 1.0]      # Light blue (Đường chéo)
            elif leak_pair_matrix[i, j] == 0:
                rgba_grid[i, j] = [0.91, 0.96, 0.91, 1.0]      # Light green (Sạch 100%)
            else:
                rgba_grid[i, j] = [1.0, 0.92, 0.93, 1.0]        # Light red (Phát hiện rò rỉ)

    ax0.imshow(rgba_grid, interpolation="nearest")
    ax0.set_xticks(range(n_splits))
    ax0.set_yticks(range(n_splits))
    ax0.set_xticklabels([s.upper() for s in splits], fontsize=11, fontweight="bold")
    ax0.set_yticklabels([s.upper() for s in splits], fontsize=11, fontweight="bold")
    ax0.set_title(f"1. Ma trận Trùng lặp dHash giữa các Split (Hamming ≤ {hamming_thresh} bit)", fontsize=12, fontweight="bold")

    for i in range(n_splits):
        for j in range(n_splits):
            if i == j:
                txt = f"{len(split_stems.get(splits[i], [])):,}\n(Tổng ảnh)"
                color = "#1D3557"
            else:
                n_pairs = leak_pair_matrix[i, j]
                if n_pairs == 0:
                    txt = "0\n(Sạch)"
                    color = "#2A9D8F"
                else:
                    n_u = leak_matrix[i, j]
                    pct = n_u / max(len(split_stems.get(splits[i], [])), 1) * 100
                    txt = f"Leak: {n_pairs:,} cặp\n({pct:.1f}% ảnh {splits[i].upper()})"
                    color = "#E63946"
            ax0.text(j, i, txt, ha="center", va="center", fontsize=10.5, fontweight="bold", color=color)

    # Panel 1: Histogram khoảng cách Test vs Train
    ax1 = axes[1]
    if min_dists is not None and len(min_dists) > 0:
        bin_width = max(1, int(round((max(min_dists) - min(min_dists)) / 30)))
        bins = np.arange(min(min_dists) - 0.5, max(min_dists) + bin_width + 0.5, bin_width)
        counts, edges, patches_list = ax1.hist(min_dists, bins=bins, edgecolor="white", linewidth=0.5, alpha=0.85)

        for edge, p in zip(edges[:-1], patches_list):
            if edge <= hamming_thresh:
                p.set_facecolor("#E63946")  # Đỏ rò rỉ
            elif edge <= hamming_thresh + 30:
                p.set_facecolor("#FB8500")  # Cam gần ngưỡng
            else:
                p.set_facecolor("#2A9D8F")  # Xanh an toàn

        ax1.axvline(hamming_thresh + 0.5, color="#E63946", linestyle="--", linewidth=2, label=f"Ngưỡng Rò rỉ (≤ {hamming_thresh})")
        ax1.axvline(np.median(min_dists), color="#1D3557", linestyle="-", linewidth=2,
                    label=f"Trung vị khoảng cách = {np.median(min_dists):.0f}")

        pct_val = dhash_test_to_train.get("near_duplicates_pct", 0)
        ax1.set_xlabel("Khoảng cách Hamming dHash giữa Test và Train (bit)", fontsize=11)
        ax1.set_ylabel("Số lượng ảnh Test", fontsize=11)
        ax1.set_title(f"2. Phân bố Độ tương đồng Test vs Train (Rò rỉ: {pct_val:.2f}%)",
                      fontsize=12, fontweight="bold")
        ax1.grid(axis="y", linestyle=":", alpha=0.5)
        ax1.legend(fontsize=10)
    else:
        ax1.text(0.5, 0.5, "Không đủ dữ liệu dHash Test/Train", ha="center", va="center")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
