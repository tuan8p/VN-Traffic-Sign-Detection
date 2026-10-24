"""Module vẽ demo ảnh mẫu kèm bounding box nhãn biển báo giao thông."""
from __future__ import annotations

from pathlib import Path
import random
from typing import Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


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
    """Figure 08 (RAW) hoặc 09 (PROCESSED): Vẽ lưới 3x3 ảnh demo kèm bounding box và tên lớp."""
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
                cname = classes.get(cid, f"cls_{cid}")
                lbl_txt = f"{cid}: {cname}"
                ax.text(
                    x1, max(0, y1 - 4), lbl_txt,
                    color="white", fontsize=8.5, fontweight="bold",
                    bbox=dict(facecolor="#E63946", alpha=0.85, pad=1.5, edgecolor="none"),
                )

            ax.set_title(f"{stem} ({len(bxs)} box)", fontsize=10.5, fontweight="bold")
            ax.axis("off")
        except Exception:
            ax.axis("off")

    for k in range(len(chosen_stems), 9):
        axes[k].axis("off")

    title_suffix = f" – {tag.upper()}" if tag else ""
    fig.suptitle(f"{fig_num:02d}. Trực quan hóa Mẫu Dữ liệu kèm Bounding Box{title_suffix}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fname = out_dir / f"{fig_num:02d}_sample_labels.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
