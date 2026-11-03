"""Streamlit demo — VN Traffic Sign Detection (BTL IPCV).

Run:  streamlit run demo/app.py

Layout:
  - Sidebar  : image input (upload / sample / drag-drop), checkpoint paths,
               score threshold, geometry controls.
  - Main     : side-by-side comparison of Faster R-CNN vs YOLO26m outputs,
               each column showing the annotated image + key metrics.
  - Bottom   : Section 3.2 — Geometric Scene Analysis (Canny + Hough Lines).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure the repo root is on the path so `vn_tsd` imports work when
# the script is run from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import io
import time

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _top_classes(boxes: list[dict], n: int = 3) -> str:
    from collections import Counter
    if not boxes:
        return "—"
    cnt = Counter(b["class_name"] for b in boxes)
    parts = [f"{name} ×{count}" for name, count in cnt.most_common(n)]
    return ", ".join(parts)


def _avg_conf(boxes: list[dict]) -> float:
    if not boxes:
        return 0.0
    return round(sum(b["score"] for b in boxes) / len(boxes), 3)


def _metrics_table(result: dict) -> list[tuple[str, str]]:
    boxes = result.get("boxes", [])
    rows = [
        ("Pipeline", result.get("pipeline", "—")),
        ("Chế độ", result.get("mode", "—")),
        ("Thời gian suy luận", f"{result.get('inference_ms', 0):.1f} ms"),
        ("Số lượng box", str(len(boxes))),
        ("Độ tin cậy TB", f"{_avg_conf(boxes):.3f}"),
        ("Top-3 nhãn dự đoán", _top_classes(boxes)),
    ]
    if result.get("mode") == "error":
        rows.append(("Lỗi", result.get("error", "")))
    return rows


def _sample_images() -> list[Path]:
    samples_dir = _REPO_ROOT / "demo" / "samples"
    if not samples_dir.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    # rglob supports both flat files and any subdirectories
    return sorted(
        [p for p in samples_dir.rglob("*") if p.suffix.lower() in exts and p.is_file()],
        key=lambda p: p.name.lower(),
    )


def _find_checkpoints(pipeline: str) -> list[Path]:
    """Scan designated folders for checkpoints.
    Prioritizes checkpoint_best.pt, followed by any other *.pt files.
    """
    candidates: list[Path] = []
    # 1. outputs/<pipeline>/...
    out_dir = _REPO_ROOT / "outputs" / pipeline
    if out_dir.is_dir():
        best_pts = sorted(out_dir.rglob("checkpoint_best.pt"))
        other_pts = sorted(p for p in out_dir.rglob("*.pt") if p.name != "checkpoint_best.pt")
        candidates.extend(best_pts + other_pts)

    # 2. Workspace root sibling directories (e.g. frcnn_seed42_best)
    parent_dir = _REPO_ROOT.parent
    keywords = ["faster_rcnn", "frcnn"] if pipeline == "faster_rcnn" else ["yolo"]
    for kw in keywords:
        for folder in parent_dir.glob(f"*{kw}*"):
            if folder.is_dir():
                best_pts = sorted(folder.rglob("checkpoint_best.pt"))
                other_pts = sorted(p for p in folder.rglob("*.pt") if p.name != "checkpoint_best.pt")
                candidates.extend(best_pts + other_pts)

    # 3. checkpoints/<pipeline>/... or checkpoints/
    ckpt_dir = _REPO_ROOT / "checkpoints" / pipeline
    if ckpt_dir.is_dir():
        candidates.extend(sorted(ckpt_dir.rglob("*.pt")))
    root_ckpt = _REPO_ROOT / "checkpoints"
    if root_ckpt.is_dir():
        candidates.extend(sorted(p for p in root_ckpt.glob("*.pt") if pipeline in p.name.lower()))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[Path] = []
    for p in candidates:
        res = str(p.resolve())
        if res not in seen and p.is_file():
            seen.add(res)
            unique.append(p)
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# Main Streamlit app
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        import streamlit as st
        from PIL import Image
    except ImportError:
        print("Install demo extras: pip install -e \".[demo]\"")
        return

    from demo.inference import predict, draw_boxes, CLASS_NAMES
    from demo.geometry import overlay_on_pil, rectify_sign_pil

    # ── Page config ────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="VN Traffic Sign Detection - Demo",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS: card style for metric boxes ─────────────────────────────────
    st.markdown("""
    <style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        border-left: 3px solid #6366f1;
    }
    .metric-label { color: #94a3b8; font-size: 12px; }
    .metric-value { color: #f8fafc; font-size: 15px; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("Cài đặt thử nghiệm")
        st.caption("Nhận diện biển báo giao thông Việt Nam (IPCV)")

        st.divider()
        st.subheader("Dữ liệu đầu vào")

        input_mode = st.radio(
            "Nguồn ảnh",
            ["Upload file", "Ảnh mẫu có sẵn"],
            horizontal=False,
        )

        pil_image: Image.Image | None = None

        def _safe_load_pil(file_bytes_or_path) -> Image.Image | None:
            """Safely load any web/phone image, auto-orient EXIF, and handle transparency."""
            from PIL import ImageOps
            try:
                if isinstance(file_bytes_or_path, (bytes, bytearray)):
                    img = Image.open(io.BytesIO(file_bytes_or_path))
                else:
                    img = Image.open(file_bytes_or_path)
                img = ImageOps.exif_transpose(img)
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    # Composite transparent regions onto clean white background
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    alpha_src = img.convert("RGBA")
                    bg.paste(alpha_src, mask=alpha_src.split()[3])
                    return bg
                return img.convert("RGB")
            except Exception as exc:
                st.error(f"Không thể mở file ảnh: {exc}")
                return None

        if input_mode == "Upload file":
            uploaded = st.file_uploader(
                "Chọn hoặc kéo thả file ảnh",
                type=["jpg", "jpeg", "png", "bmp", "webp", "jfif"],
                help="Hỗ trợ JPG, JPEG, PNG, BMP, WebP, JFIF",
            )
            if uploaded is not None:
                # Use uploaded.getvalue() to prevent EOF stream exhaustion across reruns
                pil_image = _safe_load_pil(uploaded.getvalue())

        else:  # Sample images
            samples = _sample_images()
            if not samples:
                st.warning(
                    "Chưa có ảnh mẫu trong demo/samples/.\n\n"
                    "Vui lòng copy file ảnh vào demo/samples/ rồi tải lại trang."
                )
            else:
                st.caption(f"Tìm thấy **{len(samples)}** ảnh trong `demo/samples/`")

                # State tracking for navigation
                if "sample_idx" not in st.session_state:
                    st.session_state["sample_idx"] = 0
                if st.session_state["sample_idx"] >= len(samples):
                    st.session_state["sample_idx"] = 0

                # Navigation buttons for quickly cycling through samples
                col_prev, col_next = st.columns(2)
                if col_prev.button("Ảnh trước", use_container_width=True):
                    st.session_state["sample_idx"] = (st.session_state["sample_idx"] - 1) % len(samples)
                    st.rerun()
                if col_next.button("Ảnh sau", use_container_width=True):
                    st.session_state["sample_idx"] = (st.session_state["sample_idx"] + 1) % len(samples)
                    st.rerun()

                def _fmt_sample(p: Path) -> str:
                    try:
                        rel = p.relative_to(_REPO_ROOT / "demo" / "samples")
                    except ValueError:
                        rel = p.name
                    size_kb = p.stat().st_size / 1024
                    idx = samples.index(p) + 1
                    return f"[{idx:02d}/{len(samples):02d}] {rel} ({size_kb:.1f} KB)"

                chosen = st.selectbox(
                    "Chọn từ danh sách:",
                    samples,
                    index=st.session_state["sample_idx"],
                    format_func=_fmt_sample,
                    help="Gõ tên file để lọc nhanh trong danh sách",
                )
                if chosen:
                    st.session_state["sample_idx"] = samples.index(chosen)
                    pil_image = _safe_load_pil(chosen)

        st.divider()
        st.subheader("Trọng số mô hình (Checkpoints)")

        # Faster R-CNN auto discovery
        frcnn_candidates = _find_checkpoints("faster_rcnn")
        frcnn_ckpt: Path | None = None
        if not frcnn_candidates:
            st.caption("**Faster R-CNN**: Chưa có checkpoint trong `outputs/faster_rcnn/` (chạy Mock)")
        elif len(frcnn_candidates) == 1:
            frcnn_ckpt = frcnn_candidates[0]
            try:
                rel = frcnn_ckpt.relative_to(_REPO_ROOT)
            except ValueError:
                rel = frcnn_ckpt.name
            st.caption(f"**Faster R-CNN**: Đã nhận `{rel}`")
        else:
            frcnn_ckpt = st.selectbox(
                "Faster R-CNN Checkpoint",
                frcnn_candidates,
                format_func=lambda p: str(p.relative_to(_REPO_ROOT)) if p.is_relative_to(_REPO_ROOT) else p.name,
                help="Tìm thấy nhiều checkpoint trong outputs/faster_rcnn/",
            )

        # YOLO26m auto discovery
        yolo_candidates = _find_checkpoints("yolo26m")
        yolo_ckpt: Path | None = None
        if not yolo_candidates:
            st.caption("**YOLO26m**: Chưa có checkpoint trong `outputs/yolo26m/` (chạy Mock)")
        elif len(yolo_candidates) == 1:
            yolo_ckpt = yolo_candidates[0]
            try:
                rel = yolo_ckpt.relative_to(_REPO_ROOT)
            except ValueError:
                rel = yolo_ckpt.name
            st.caption(f"**YOLO26m**: Đã nhận `{rel}`")
        else:
            yolo_ckpt = st.selectbox(
                "YOLO26m Checkpoint",
                yolo_candidates,
                format_func=lambda p: str(p.relative_to(_REPO_ROOT)) if p.is_relative_to(_REPO_ROOT) else p.name,
                help="Tìm thấy nhiều checkpoint trong outputs/yolo26m/",
            )

        st.divider()
        st.subheader("Tham số mô hình")

        score_thresh = st.slider(
            "Ngưỡng tin cậy (Score threshold)", 0.05, 0.95, 0.25, 0.05,
            help="Chỉ hiển thị box có điểm tin cậy lớn hơn hoặc bằng ngưỡng",
        )

        has_torch = False
        has_cuda = False
        try:
            import torch
            has_torch = True
            has_cuda = torch.cuda.is_available()
        except Exception:
            pass

        if not has_torch and (frcnn_ckpt is not None or yolo_ckpt is not None):
            st.error(
                "Cảnh báo môi trường: Không tìm thấy thư viện PyTorch trong Python đang chạy. "
                "Vui lòng chạy Streamlit bằng Python 3.12 của hệ thống: "
                "`python -m streamlit run demo/app.py`"
            )

        device_options = ["cpu", "cuda"]
        device_labels = {
            "cpu": "CPU",
            "cuda": f"CUDA GPU {'(Sẵn sàng)' if has_cuda else '(Không khả dụng - dùng CPU)'}",
        }
        device = st.radio(
            "Thiết bị tính toán (Device)",
            device_options,
            index=1 if has_cuda else 0,
            format_func=lambda d: device_labels[d],
            horizontal=True,
            help="Chỉ chọn CPU hoặc CUDA",
        )

        st.divider()
        st.subheader("Phân tích hình học (Mục 3.2)")
        show_geometry = st.toggle("Bật phân tích Canny + Hough", value=True)
        canny_lo = st.slider("Canny low threshold", 10, 200, 50, 5)
        canny_hi = st.slider("Canny high threshold", 50, 400, 150, 10)
        hough_thresh = st.slider("Hough threshold", 20, 200, 80, 10)
        geo_mode = st.radio(
            "Chế độ hiển thị overlay",
            ["Edges", "Lines", "Combined"],
            index=2,
            horizontal=True,
        )
        geo_key_map = {
            "Edges": "edge_overlay",
            "Lines": "line_overlay",
            "Combined": "combined_overlay",
        }

        run_btn = st.button("Chạy phân tích", type="primary", use_container_width=True)

    # ── Main panel ─────────────────────────────────────────────────────────────
    st.title("Nhận diện biển báo giao thông Việt Nam")
    st.caption(
        "So sánh mô hình **Faster R-CNN (ResNet-50)** và **YOLO26m** · "
        "Phân tích cấu trúc hình học Mục 3.2 · BTL IPCV"
    )

    if pil_image is None:
        st.info(
            "Chọn ảnh đầu vào ở thanh bên (sidebar) để bắt đầu.\n\n"
            "Có thể upload file, kéo thả hoặc chọn từ danh sách ảnh mẫu."
        )
        st.stop()

    # Show original image
    with st.expander("Ảnh gốc", expanded=True):
        w_orig, h_orig = pil_image.size
        st.image(pil_image, caption=f"Kích thước gốc: {w_orig}×{h_orig} px", use_column_width=False, width=min(w_orig, 640))

    st.divider()

    # ── Run inference when button pressed OR parameters change ────────────────
    img_sig = (pil_image.size, pil_image.tobytes()[:256]) if pil_image is not None else None
    current_run_key = (img_sig, str(frcnn_ckpt), str(yolo_ckpt), score_thresh, device)
    current_geo_key = (img_sig, show_geometry, canny_lo, canny_hi, hough_thresh)

    need_inference = run_btn or (st.session_state.get("last_run_key") != current_run_key)
    need_geometry = run_btn or (st.session_state.get("last_geo_key") != current_geo_key)

    if need_inference:
        st.session_state["last_run_key"] = current_run_key

        with st.spinner("Đang chạy suy luận mô hình..."):
            frcnn_result = predict(
                pil_image,
                checkpoint=frcnn_ckpt,
                pipeline="faster_rcnn",
                device=device,
                score_threshold=score_thresh,
            )
            yolo_result = predict(
                pil_image,
                checkpoint=yolo_ckpt,
                pipeline="yolo26m",
                device=device,
                score_threshold=score_thresh,
            )

        st.session_state["frcnn_result"] = frcnn_result
        st.session_state["yolo_result"] = yolo_result

    if need_geometry:
        st.session_state["last_geo_key"] = current_geo_key
        if show_geometry:
            with st.spinner("Đang phân tích hình học..."):
                geo_result = analyze_scene_cached(
                    pil_image, canny_lo, canny_hi, hough_thresh
                )
            st.session_state["geo_result"] = geo_result
        else:
            st.session_state["geo_result"] = None

    frcnn_result = st.session_state.get("frcnn_result")
    yolo_result = st.session_state.get("yolo_result")
    geo_result = st.session_state.get("geo_result")

    if frcnn_result is None or yolo_result is None:
        st.stop()

    # ── Side-by-side model comparison ─────────────────────────────────────────
    st.subheader("So sánh hai mô hình")

    frcnn_mode = frcnn_result.get("mode", "mock")
    yolo_mode = yolo_result.get("mode", "mock")

    if frcnn_mode == "mock" or yolo_mode == "mock":
        st.warning(
            "Một hoặc cả hai mô hình đang chạy ở chế độ giả lập (Mock) do chưa có file checkpoint. "
            "Khi có checkpoint trong thư mục outputs/, hệ thống sẽ tự động chuyển sang chế độ thực."
        )

    col_frcnn, col_yolo = st.columns(2, gap="large")

    # ── Faster R-CNN column ───────────────────────────────────────────────────
    with col_frcnn:
        st.markdown("### Faster R-CNN (ResNet-50)")
        badge = "Chế độ: Mock" if frcnn_mode == "mock" else ("Chế độ: Checkpoint thực" if frcnn_mode == "real" else "Trạng thái: Lỗi")
        st.caption(badge)

        frcnn_annotated = draw_boxes(pil_image, frcnn_result)
        st.image(frcnn_annotated, caption="Kết quả Faster R-CNN", use_column_width=True)

        st.markdown("**Chỉ số chi tiết**")
        for label, value in _metrics_table(frcnn_result):
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── YOLO26m column ────────────────────────────────────────────────────────
    with col_yolo:
        st.markdown("### YOLO26m")
        badge = "Chế độ: Mock" if yolo_mode == "mock" else ("Chế độ: Checkpoint thực" if yolo_mode == "real" else "Trạng thái: Lỗi")
        st.caption(badge)

        yolo_annotated = draw_boxes(pil_image, yolo_result)
        st.image(yolo_annotated, caption="Kết quả YOLO26m", use_column_width=True)

        st.markdown("**Chỉ số chi tiết**")
        for label, value in _metrics_table(yolo_result):
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Comparison summary table ───────────────────────────────────────────────
    st.divider()
    st.subheader("Bảng so sánh tổng hợp")

    fboxes = frcnn_result.get("boxes", [])
    yboxes = yolo_result.get("boxes", [])

    import pandas as pd
    summary = pd.DataFrame({
        "": ["Số box phát hiện", "Độ tin cậy trung bình", "Thời gian suy luận (ms)", "Nhãn xuất hiện nhiều nhất"],
        "Faster R-CNN": [
            len(fboxes),
            f"{_avg_conf(fboxes):.3f}",
            f"{frcnn_result.get('inference_ms', 0):.1f}",
            _top_classes(fboxes, n=1),
        ],
        "YOLO26m": [
            len(yboxes),
            f"{_avg_conf(yboxes):.3f}",
            f"{yolo_result.get('inference_ms', 0):.1f}",
            _top_classes(yboxes, n=1),
        ],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # ── Section 3.2 — Geometry ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Phân tích hình học và cấu trúc cảnh (Mục 3.2)")
    st.caption(
        "Vận dụng kiến thức hình học thị giác máy tính: (A) Trích xuất cấu trúc biên và đường thẳng toàn cảnh; "
        "(B) Ước lượng phép biến đổi phối cảnh 2D Homography để nắn thẳng (rectify) biển báo bị chụp nghiêng."
    )

    tab_global, tab_rectify = st.tabs([
        "3.2.A: Cấu trúc biên & đường thẳng (Canny + Hough)",
        "3.2.B: Nắn thẳng phối cảnh biển báo (Homography Rectification)",
    ])

    with tab_global:
        if not show_geometry:
            st.info("Bật tùy chọn 'Phân tích Canny + Hough' ở thanh bên để xem kết quả toàn cảnh.")
        elif geo_result is None:
            st.warning("Chưa có kết quả phân tích. Nhấn 'Chạy phân tích' để thực hiện.")
        else:
            geo_overlay_key = geo_key_map[geo_mode]
            geo_img_bgr = geo_result[geo_overlay_key]

            # Convert BGR -> RGB for display
            import cv2
            geo_img_rgb = cv2.cvtColor(geo_img_bgr, cv2.COLOR_BGR2RGB)
            from PIL import Image as PILImage
            geo_pil = PILImage.fromarray(geo_img_rgb)

            col_orig_geo, col_geo_out = st.columns(2, gap="medium")
            with col_orig_geo:
                st.markdown("**Ảnh gốc**")
                st.image(pil_image, use_column_width=True)
            with col_geo_out:
                st.markdown(f"**Kết quả overlay ({geo_mode})**")
                st.image(geo_pil, use_column_width=True)

            # Geometry stats
            stats = geo_result.get("stats", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Số pixel biên", f"{stats.get('num_edge_px', 0):,}")
            c2.metric("Mật độ biên", f"{stats.get('edge_density_pct', 0):.1f}%")
            c3.metric("Số đoạn thẳng", str(stats.get("num_lines", 0)))
            c4.metric("Độ dài TB đoạn thẳng", f"{stats.get('mean_line_length', 0):.1f} px")

            with st.expander("Thông số thuật toán Canny / Hough"):
                params_df = pd.DataFrame({
                    "Tham số": ["Canny Low", "Canny High", "Hough Threshold",
                                 "Min Line Length", "Max Line Gap", "Blur Kernel"],
                    "Giá trị": [
                        stats.get("canny_lo", canny_lo),
                        stats.get("canny_hi", canny_hi),
                        stats.get("hough_threshold", hough_thresh),
                        40, 10, 5,
                    ],
                })
                st.dataframe(params_df, use_container_width=True, hide_index=True)

    with tab_rectify:
        st.markdown(
            "Phép biến đổi phối cảnh 2D Homography cho phép ánh xạ các tọa độ biển báo bị méo/nghiêng góc "
            "về mặt phẳng chuẩn chính diện (frontal view), giúp quan sát và nhận dạng nội dung rõ nét."
        )

        # Tập hợp tất cả các box phát hiện được
        all_detected = []
        for b in frcnn_result.get("boxes", []):
            all_detected.append(("Faster R-CNN", b))
        for b in yolo_result.get("boxes", []):
            all_detected.append(("YOLO26m", b))

        padding_pct = 0.15
        if not all_detected:
            st.info(
                "Chưa có biển báo nào vượt qua ngưỡng tin cậy để tự động nắn thẳng. "
                "Đang kích hoạt chế độ nắn thẳng thử nghiệm trên vùng trung tâm ảnh."
            )
            w_img, h_img = pil_image.size
            chosen_box = [w_img * 0.25, h_img * 0.25, w_img * 0.75, h_img * 0.75]
            chosen_label = "Vùng thử nghiệm trung tâm ảnh"
            chosen_score = 1.0
        else:
            col_sel1, col_sel2 = st.columns([3, 1])
            with col_sel1:
                selected_idx = st.selectbox(
                    "Chọn biển báo cần nắn thẳng:",
                    range(len(all_detected)),
                    format_func=lambda i: f"[{all_detected[i][0]}] #{i+1}: {all_detected[i][1]['class_name']} (độ tin cậy: {all_detected[i][1]['score']:.2f})",
                )
                chosen_source, chosen_box_dict = all_detected[selected_idx]
                chosen_box = chosen_box_dict["xyxy"]
                chosen_label = f"{chosen_box_dict['class_name']} ({chosen_source})"
                chosen_score = chosen_box_dict["score"]
            with col_sel2:
                padding_pct = st.slider("Padding ROI (%)", 5, 40, 15, 5, help="Mở rộng biên xung quanh box để không mất viền biển") / 100.0

        try:
            rect_res = rectify_sign_pil(pil_image, chosen_box, padding=padding_pct)

            col_r1, col_r2, col_r3 = st.columns([1, 1, 1.2], gap="medium")
            with col_r1:
                st.markdown("**Vùng ROI gốc (với 4 đỉnh nguồn)**")
                st.image(rect_res["roi_annotated_pil"], caption=f"Vùng cắt gốc: {chosen_label}", use_column_width=True)
            with col_r2:
                st.markdown("**Biển báo sau khi nắn thẳng (Frontal View)**")
                st.image(rect_res["rectified_pil"], caption="Kích thước chuẩn: 240×240 px", use_column_width=True)
            with col_r3:
                st.markdown("**Ma trận Homography $\\mathbf{H}_{3 \\times 3}$**")
                H_mat = rect_res["homography_matrix"]
                h_df = pd.DataFrame(
                    H_mat,
                    columns=["h1", "h2", "h3"],
                    index=["h1x", "h2x", "h3x"],
                )
                st.dataframe(h_df.style.format("{:.4e}"), use_container_width=True)
                st.caption(f"Phương pháp trích xuất 4 góc: **{rect_res['method']}**")

                with st.expander("Công thức toán học phép biến đổi"):
                    st.latex(r"""
                    \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
                    = \begin{bmatrix}
                    h_{11} & h_{12} & h_{13} \\
                    h_{21} & h_{22} & h_{23} \\
                    h_{31} & h_{32} & h_{33}
                    \end{bmatrix}
                    \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
                    """)
                    st.caption("Ma trận Homography được tính toán bằng thuật toán DLT (Direct Linear Transformation) từ 4 cặp điểm tương ứng nguồn - đích.")
        except Exception as e:
            st.warning(f"Không thể thực hiện nắn thẳng cho vùng chọn: {e}")

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "IPCV BTL — Nhóm demo | Faster R-CNN (ResNet-50) & YOLO26m | "
        "Section 3.2: Canny + HoughLinesP & 2D Homography Rectification | Section 3.4: Streamlit interactive demo"
    )


# ── Cached geometry analysis (avoid rerunning on every widget interaction) ────

def analyze_scene_cached(pil_image, canny_lo, canny_hi, hough_thresh):
    """Run geometry analysis; plain function (Streamlit caching applied at call site)."""
    from demo.geometry import analyze_scene
    import cv2
    import numpy as np

    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return analyze_scene(
        bgr,
        canny_lo=canny_lo,
        canny_hi=canny_hi,
        hough_threshold=hough_thresh,
    )


if __name__ == "__main__":
    main()
