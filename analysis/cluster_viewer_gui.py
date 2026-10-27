"""dHash Cluster Viewer GUI - Ứng dụng xem cụm ảnh tương đồng trên Desktop.

Giao diện Windows Tkinter tương tác trực tiếp:
- Xem danh sách toàn bộ các cụm ảnh (tìm kiếm, sắp xếp, lọc cụm đa ảnh).
- Hiển thị toàn bộ ảnh trong từng cụm dạng lưới (Grid).
- Bật/tắt Bounding Box nhãn YOLO và tên biển báo tiếng Việt.
- Bấm/Click đúp vào ảnh để xem ảnh phóng to chi tiết kích thước lớn.
- Chuyển cụm nhanh bằng phím mũi tên [◄], [►] hoặc nút bấm.

Chạy:
    python cluster_viewer_gui.py
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import csv
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont


class ClusterViewerApp:
    def __init__(self, root: tk.Tk, data_root: str = "data/raw", clusters_json: str = "analysis/figures/raw/dhash_clusters.json"):
        self.root = root
        self.root.title("dHash Cluster Viewer — Phân tích Cụm Ảnh Dashcam")
        self.root.geometry("1300x820")
        self.root.minsize(1000, 650)

        # Style & Font
        self._setup_styles()

        # Paths
        self.data_root = Path(data_root).resolve()
        self.clusters_json_path = Path(clusters_json).resolve()
        self.classes_csv_path = Path("configs/classes.csv").resolve()

        # State
        self.classes: dict[int, str] = {}
        self.stem_to_cluster: dict[str, int] = {}
        self.cluster_to_stems: dict[int, list[str]] = defaultdict(list)
        self.img_map: dict[str, Path] = {}
        self.current_cluster_id: int | None = None
        self.sorted_cluster_ids: list[int] = []
        self.display_cluster_ids: list[int] = []
        self.metadata: dict = {}

        # Cache thumbnail ImageTk
        self._thumb_cache: dict[tuple[str, bool], ImageTk.PhotoImage] = {}
        self._current_photo_refs: list[ImageTk.PhotoImage] = []

        # UI variables
        self.var_draw_boxes = tk.BooleanVar(value=True)
        self.var_search = tk.StringVar(value="")
        self.var_filter = tk.StringVar(value="Cụm đa ảnh (≥ 2 ảnh)")
        self.var_sort = tk.StringVar(value="Số ảnh giảm dần")

        # Build UI
        self._build_menu()
        self._build_layout()

        # Load data
        self._load_classes()
        self._load_dataset_images()
        self._load_clusters(self.clusters_json_path)

        # Keyboard shortcuts
        self.root.bind("<Left>", lambda e: self._on_prev_cluster())
        self.root.bind("<Right>", lambda e: self._on_next_cluster())
        self.root.bind("<Up>", lambda e: self._on_arrow_cluster(-1))
        self.root.bind("<Down>", lambda e: self._on_arrow_cluster(1))

    # ──────────────────────────────────────────────────────────────────────────
    # Styles & Colors
    # ──────────────────────────────────────────────────────────────────────────
    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self.c_bg = "#F4F6F9"
        self.c_card = "#FFFFFF"
        self.c_primary = "#1D3557"
        self.c_accent = "#E63946"
        self.c_accent_blue = "#457B9D"
        self.c_text = "#2B2D42"
        self.c_subtext = "#6C757D"
        self.c_border = "#DEE2E6"

        self.root.configure(bg=self.c_bg)

        style.configure("Treeview", font=("Segoe UI", 10), rowheight=26)
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", self.c_accent_blue)], foreground=[("selected", "white")])

        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), background=self.c_primary, foreground="white")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), background=self.c_accent, foreground="white")

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tệp (File)", menu=file_menu)
        file_menu.add_command(label="Mở file cụm dhash_clusters.json...", command=self._on_open_json)
        file_menu.add_command(label="Đổi thư mục dữ liệu (data_root)...", command=self._on_open_dataroot)
        file_menu.add_separator()
        file_menu.add_command(label="Thoát", command=self.root.quit)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Trợ giúp", menu=help_menu)
        help_menu.add_command(label="Phím tắt & Hướng dẫn", command=self._show_help)

    def _build_layout(self):
        # ── Top Status Bar ──
        self.top_bar = tk.Frame(self.root, bg=self.c_primary, height=44)
        self.top_bar.pack(side=tk.TOP, fill=tk.X)
        self.top_bar.pack_propagate(False)

        self.lbl_header = tk.Label(
            self.top_bar,
            text="Bộ Trực Quan Hóa Cụm Ảnh Tương Đồng (dHash Cluster Viewer)",
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg=self.c_primary,
            padx=15,
        )
        self.lbl_header.pack(side=tk.LEFT, fill=tk.Y)

        self.lbl_stats = tk.Label(
            self.top_bar,
            text="Đang tải dữ liệu...",
            font=("Segoe UI", 10),
            fg="#A8DADC",
            bg=self.c_primary,
            padx=15,
        )
        self.lbl_stats.pack(side=tk.RIGHT, fill=tk.Y)

        # ── Main Container (Split Left / Right) ──
        self.main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bg=self.c_border, sashwidth=5)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        # ── LEFT PANEL: Cluster Explorer ──
        self.left_frame = tk.Frame(self.main_paned, bg=self.c_bg, width=380, padx=10, pady=10)
        self.left_frame.pack_propagate(False)
        self.main_paned.add(self.left_frame)

        # Controls & Search in Left Panel
        ctrl_frame = tk.Frame(self.left_frame, bg=self.c_bg)
        ctrl_frame.pack(fill=tk.X, pady=(0, 8))

        lbl_search = tk.Label(ctrl_frame, text="🔍 Tìm Cụm ID hoặc Tên ảnh:", font=("Segoe UI", 9, "bold"), bg=self.c_bg, fg=self.c_text)
        lbl_search.pack(anchor=tk.W)

        self.entry_search = ttk.Entry(ctrl_frame, textvariable=self.var_search, font=("Segoe UI", 10))
        self.entry_search.pack(fill=tk.X, pady=(2, 6))
        self.entry_search.bind("<KeyRelease>", lambda e: self._apply_filter())

        # Filter & Sort row
        row_filter = tk.Frame(ctrl_frame, bg=self.c_bg)
        row_filter.pack(fill=tk.X, pady=2)

        lbl_f = tk.Label(row_filter, text="Lọc:", font=("Segoe UI", 9), bg=self.c_bg, fg=self.c_text)
        lbl_f.pack(side=tk.LEFT, padx=(0, 4))
        self.cb_filter = ttk.Combobox(
            row_filter,
            textvariable=self.var_filter,
            values=["Cụm đa ảnh (≥ 2 ảnh)", "Cụm lớn (≥ 5 ảnh)", "Tất cả các cụm (cả 1 ảnh)"],
            state="readonly",
            width=20,
            font=("Segoe UI", 9),
        )
        self.cb_filter.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cb_filter.bind("<<ComboboxSelected>>", lambda e: self._apply_filter())

        # Treeview Cluster List
        tree_frame = tk.Frame(self.left_frame, bg=self.c_bg)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        columns = ("cid", "count", "preview")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("cid", text="Cụm ID", anchor=tk.CENTER)
        self.tree.heading("count", text="Số ảnh", anchor=tk.CENTER)
        self.tree.heading("preview", text="Mẫu file ảnh", anchor=tk.W)

        self.tree.column("cid", width=70, anchor=tk.CENTER, stretch=False)
        self.tree.column("count", width=65, anchor=tk.CENTER, stretch=False)
        self.tree.column("preview", width=180, anchor=tk.W)

        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Bottom Left Controls
        bot_ctrl = tk.Frame(self.left_frame, bg=self.c_bg)
        bot_ctrl.pack(fill=tk.X, pady=4)

        chk_box = tk.Checkbutton(
            bot_ctrl,
            text="Vẽ Bounding Box & Tên biển",
            variable=self.var_draw_boxes,
            command=self._on_toggle_boxes,
            font=("Segoe UI", 10, "bold"),
            bg=self.c_bg,
            fg=self.c_primary,
            activebackground=self.c_bg,
        )
        chk_box.pack(anchor=tk.W, pady=(0, 6))

        nav_box = tk.Frame(bot_ctrl, bg=self.c_bg)
        nav_box.pack(fill=tk.X)

        btn_prev = tk.Button(
            nav_box, text="◄ Cụm Trước", command=self._on_prev_cluster,
            font=("Segoe UI", 9, "bold"), bg="#E0E7EF", fg=self.c_primary, relief=tk.GROOVE, padx=8, pady=4
        )
        btn_prev.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        btn_next = tk.Button(
            nav_box, text="Cụm Sau ►", command=self._on_next_cluster,
            font=("Segoe UI", 9, "bold"), bg=self.c_accent_blue, fg="white", relief=tk.GROOVE, padx=8, pady=4
        )
        btn_next.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # ── RIGHT PANEL: Image Gallery / Frame ──
        self.right_frame = tk.Frame(self.main_paned, bg=self.c_card, padx=12, pady=10)
        self.main_paned.add(self.right_frame)

        # Right Header
        self.r_header = tk.Frame(self.right_frame, bg=self.c_card)
        self.r_header.pack(fill=tk.X, pady=(0, 8))

        self.lbl_cluster_title = tk.Label(
            self.r_header,
            text="Chọn một cụm từ danh sách bên trái để hiển thị toàn bộ ảnh",
            font=("Segoe UI", 13, "bold"),
            fg=self.c_primary,
            bg=self.c_card,
        )
        self.lbl_cluster_title.pack(side=tk.LEFT, anchor=tk.W)

        self.btn_open_folder = tk.Button(
            self.r_header,
            text="📂 Mở Thư Mục Ảnh",
            font=("Segoe UI", 9),
            bg=self.c_bg,
            command=self._on_open_current_folder,
            relief=tk.GROOVE,
            padx=8,
            pady=3,
        )
        self.btn_open_folder.pack(side=tk.RIGHT)

        # Scrollable Image Canvas
        canvas_container = tk.Frame(self.right_frame, bg=self.c_card)
        canvas_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_container, bg=self.c_card, highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)

        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.gallery_frame = tk.Frame(self.canvas, bg=self.c_card)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.gallery_frame, anchor=tk.NW)

        self.gallery_frame.bind("<Configure>", self._on_gallery_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    # ──────────────────────────────────────────────────────────────────────────
    # Data Loading
    # ──────────────────────────────────────────────────────────────────────────
    def _load_classes(self):
        self.classes.clear()
        if self.classes_csv_path.exists():
            try:
                with open(self.classes_csv_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self.classes[int(row["class_id"])] = row["name_vi"]
            except Exception as e:
                print(f"[!] Lỗi đọc classes.csv: {e}")

    def _load_dataset_images(self):
        self.img_map.clear()
        img_dir = self.data_root / "images"
        if img_dir.is_dir():
            for p in img_dir.glob("*.*"):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    self.img_map[p.stem] = p
        print(f"[GUI] Đã nạp {len(self.img_map)} ảnh từ {img_dir}")

    def _load_clusters(self, json_path: Path):
        data = None
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                messagebox.showerror("Lỗi đọc JSON", f"Không thể đọc file {json_path}:\n{e}")
                return
        else:
            img_paths = list(self.img_map.values())
            if img_paths:
                print(f"[GUI] File {json_path.name} không tồn tại. Đang tự động gom cụm dHash trực tiếp từ {len(img_paths)} ảnh...")
                from analysis.helpers.clustering import cluster_images_dhash
                data = cluster_images_dhash(img_paths, hash_size=32, hamming_thresh=380)
            else:
                messagebox.showwarning(
                    "Chưa có dữ liệu ảnh",
                    f"Không tìm thấy ảnh tại {self.data_root / 'images'} để gom cụm."
                )
                return

        self.metadata = data
        self.stem_to_cluster = data.get("stem_to_cluster", {})

        self.cluster_to_stems.clear()
        for stem, cid in self.stem_to_cluster.items():
            self.cluster_to_stems[int(cid)].append(stem)

        # Sắp xếp stems trong mỗi cụm theo tên
        for cid in self.cluster_to_stems:
            self.cluster_to_stems[cid].sort()

        # Update stats banner
        hsize = data.get("hash_size", "?")
        hdist = data.get("hamming_threshold", "?")
        n_tot = data.get("total_clusters", len(self.cluster_to_stems))
        n_multi = data.get("multi_image_clusters", sum(1 for s in self.cluster_to_stems.values() if len(s) >= 2))
        n_imgs = data.get("total_images", len(self.stem_to_cluster))

        self.lbl_stats.config(
            text=f"dHash: size={hsize}, hamming={hdist} | {n_imgs} ảnh | {n_tot} cụm ({n_multi} cụm đa ảnh)"
        )

        self._apply_filter()

    # ──────────────────────────────────────────────────────────────────────────
    # Filtering & Listbox Updates
    # ──────────────────────────────────────────────────────────────────────────
    def _apply_filter(self):
        search = self.var_search.get().strip().lower()
        filter_mode = self.var_filter.get()

        # Lọc danh sách cụm
        filtered_cids = []
        for cid, stems in self.cluster_to_stems.items():
            count = len(stems)

            # Filter combobox
            if filter_mode == "Cụm đa ảnh (≥ 2 ảnh)" and count < 2:
                continue
            elif filter_mode == "Cụm lớn (≥ 5 ảnh)" and count < 5:
                continue

            # Search entry
            if search:
                cid_match = search in str(cid)
                stem_match = any(search in s.lower() for s in stems)
                if not (cid_match or stem_match):
                    continue

            filtered_cids.append(cid)

        # Sắp xếp theo số ảnh giảm dần
        filtered_cids.sort(key=lambda cid: (len(self.cluster_to_stems[cid]), -cid), reverse=True)
        self.display_cluster_ids = filtered_cids

        # Cập nhật Treeview
        self.tree.delete(*self.tree.get_children())
        for cid in self.display_cluster_ids:
            stems = self.cluster_to_stems[cid]
            preview = ", ".join(stems[:3]) + ("..." if len(stems) > 3 else "")
            self.tree.insert("", tk.END, iid=str(cid), values=(f"#{cid}", len(stems), preview))

        # Tự động chọn cụm đầu tiên nếu có
        if self.display_cluster_ids:
            first_id = str(self.display_cluster_ids[0])
            self.tree.selection_set(first_id)
            self.tree.see(first_id)
            self._render_cluster(self.display_cluster_ids[0])
        else:
            self._clear_gallery("Không có cụm nào thỏa mãn điều kiện tìm kiếm/lọc.")

    def _on_tree_select(self, event):
        selected = self.tree.selection()
        if not selected:
            return
        cid = int(selected[0])
        if cid != self.current_cluster_id:
            self._render_cluster(cid)

    def _on_arrow_cluster(self, delta: int):
        selected = self.tree.selection()
        if not selected:
            return
        cid = int(selected[0])
        if cid in self.display_cluster_ids:
            idx = self.display_cluster_ids.index(cid)
            new_idx = max(0, min(len(self.display_cluster_ids) - 1, idx + delta))
            new_cid = str(self.display_cluster_ids[new_idx])
            self.tree.selection_set(new_cid)
            self.tree.see(new_cid)
            self._render_cluster(self.display_cluster_ids[new_idx])

    def _on_prev_cluster(self):
        self._on_arrow_cluster(-1)

    def _on_next_cluster(self):
        self._on_arrow_cluster(1)

    def _on_toggle_boxes(self):
        if self.current_cluster_id is not None:
            self._render_cluster(self.current_cluster_id)

    # ──────────────────────────────────────────────────────────────────────────
    # Gallery Rendering
    # ──────────────────────────────────────────────────────────────────────────
    def _clear_gallery(self, message: str = ""):
        for widget in self.gallery_frame.winfo_children():
            widget.destroy()
        self._current_photo_refs.clear()
        if message:
            self.lbl_cluster_title.config(text=message)

    def _render_cluster(self, cluster_id: int):
        self.current_cluster_id = cluster_id
        stems = self.cluster_to_stems.get(cluster_id, [])
        n_imgs = len(stems)

        self.lbl_cluster_title.config(
            text=f"CỤM #{cluster_id:04d} — {n_imgs} ảnh liên tiếp / cùng phân cảnh dashcam"
        )

        self._clear_gallery()

        # Canvas width để tính số cột lưới
        canvas_w = max(self.canvas.winfo_width(), 600)
        card_w = 340
        num_cols = max(1, min(4, canvas_w // card_w))

        lbl_dir = self.data_root / "labels"
        draw_boxes = self.var_draw_boxes.get()

        for idx, stem in enumerate(stems):
            row = idx // num_cols
            col = idx % num_cols

            card = tk.Frame(self.gallery_frame, bg="#FFFFFF", bd=1, relief=tk.SOLID, padx=6, pady=6)
            card.grid(row=row, column=col, padx=8, pady=8, sticky=tk.NSEW)

            img_p = self.img_map.get(stem)
            boxes = self._load_yolo_boxes(lbl_dir / f"{stem}.txt")

            # Tạo thumbnail
            photo = self._get_thumbnail(img_p, boxes if draw_boxes else [], target_w=320, target_h=210)
            if photo:
                self._current_photo_refs.append(photo)
                img_lbl = tk.Label(card, image=photo, bg="#FFFFFF", cursor="hand2")
                img_lbl.pack()
                # Click hoặc click đúp để xem ảnh phóng to full HD
                img_lbl.bind("<Button-1>", lambda e, p=img_p, b=boxes: self._show_full_image(p, b))
            else:
                placeholder = tk.Label(card, text="[Không tìm thấy ảnh]", width=35, height=10, bg="#E9ECEF")
                placeholder.pack()

            # Thông tin nhãn và file
            n_box = len(boxes)
            info_txt = f"#{idx+1}: {stem}.jpg\n({n_box} nhãn biển báo)"
            cap_lbl = tk.Label(
                card, text=info_txt, font=("Segoe UI", 9, "bold"), fg=self.c_primary, bg="#FFFFFF", justify=tk.CENTER
            )
            cap_lbl.pack(pady=(4, 2))

        # Reset cuộn về đỉnh
        self.canvas.yview_moveto(0)
        self.gallery_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _get_thumbnail(self, img_path: Path | None, boxes: list, target_w: int = 320, target_h: int = 210) -> ImageTk.PhotoImage | None:
        if not img_path or not img_path.exists():
            return None

        cache_key = (f"{img_path.stem}_{len(boxes)}_{self.var_draw_boxes.get()}_{target_w}_{target_h}", bool(boxes))
        if cache_key in self._thumb_cache:
            return self._thumb_cache[cache_key]

        try:
            with Image.open(img_path) as im:
                orig_w, orig_h = im.size
                im = im.convert("RGB")

                # Scale thumbnail
                im.thumbnail((target_w, target_h), Image.Resampling.BILINEAR)
                tw, th = im.size

                if boxes and self.var_draw_boxes.get():
                    draw = ImageDraw.Draw(im)
                    sx = tw / orig_w
                    sy = th / orig_h

                    for cid, cx, cy, bw, bh in boxes:
                        x1 = (cx - bw / 2) * orig_w * sx
                        y1 = (cy - bh / 2) * orig_h * sy
                        x2 = (cx + bw / 2) * orig_w * sx
                        y2 = (cy + bh / 2) * orig_h * sy

                        draw.rectangle([x1, y1, x2, y2], outline="#E63946", width=2)
                        cname = self.classes.get(cid, f"Lớp {cid}")
                        tag = f"[{cid}] {cname[:12]}"
                        # Label background
                        draw.rectangle([x1, max(0, y1 - 14), min(tw, x1 + len(tag) * 6 + 8), y1], fill="#E63946")
                        draw.text((x1 + 2, max(0, y1 - 13)), tag, fill="white")

                photo = ImageTk.PhotoImage(im)
                self._thumb_cache[cache_key] = photo
                return photo
        except Exception as e:
            print(f"[!] Lỗi tạo thumbnail {img_path}: {e}")
            return None

    def _load_yolo_boxes(self, txt_path: Path) -> list[tuple[int, float, float, float, float]]:
        boxes = []
        if txt_path.exists():
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:5])
                            boxes.append((cid, cx, cy, bw, bh))
            except Exception:
                pass
        return boxes

    # ──────────────────────────────────────────────────────────────────────────
    # Popup xem ảnh phóng to chi tiết
    # ──────────────────────────────────────────────────────────────────────────
    def _show_full_image(self, img_path: Path | None, boxes: list):
        if not img_path or not img_path.exists():
            return

        win = tk.Toplevel(self.root)
        win.title(f"Xem Chi Tiết: {img_path.name}")
        win.geometry("1100x780")
        win.configure(bg="#1E1E2E")

        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                orig_w, orig_h = im.size

                # Resize để vừa khung xem 1040x680
                max_w, max_h = 1040, 680
                scale = min(max_w / orig_w, max_h / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)

                im_resized = im.resize((new_w, new_h), Image.Resampling.BILINEAR)

                draw = ImageDraw.Draw(im_resized)
                for cid, cx, cy, bw, bh in boxes:
                    x1 = (cx - bw / 2) * orig_w * scale
                    y1 = (cy - bh / 2) * orig_h * scale
                    x2 = (cx + bw / 2) * orig_w * scale
                    y2 = (cy + bh / 2) * orig_h * scale

                    draw.rectangle([x1, y1, x2, y2], outline="#E63946", width=3)
                    cname = self.classes.get(cid, f"Lớp {cid}")
                    tag = f"[{cid}] {cname}"
                    draw.rectangle([x1, max(0, y1 - 20), min(new_w, x1 + len(tag) * 8 + 12), y1], fill="#E63946")
                    draw.text((x1 + 4, max(0, y1 - 18)), tag, fill="white")

                photo_full = ImageTk.PhotoImage(im_resized)
                lbl = tk.Label(win, image=photo_full, bg="#1E1E2E")
                lbl.image = photo_full
                lbl.pack(pady=10)

                lbl_info = tk.Label(
                    win,
                    text=f"{img_path.name} | Gốc: {orig_w}×{orig_h}px | {len(boxes)} biển báo phát hiện",
                    font=("Segoe UI", 11, "bold"),
                    fg="white",
                    bg="#1E1E2E",
                )
                lbl_info.pack(pady=5)
        except Exception as e:
            messagebox.showerror("Lỗi mở ảnh", f"Không mở được ảnh:\n{e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Canvas Scrolling & Resizing
    # ──────────────────────────────────────────────────────────────────────────
    def _on_gallery_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        if self.canvas.winfo_height() < self.gallery_frame.winfo_height():
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ──────────────────────────────────────────────────────────────────────────
    # Menu Commands
    # ──────────────────────────────────────────────────────────────────────────
    def _on_open_json(self):
        f = filedialog.askopenfilename(
            title="Chọn file dhash_clusters.json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            initialdir=str(self.clusters_json_path.parent),
        )
        if f:
            self.clusters_json_path = Path(f)
            self._thumb_cache.clear()
            self._load_clusters(self.clusters_json_path)

    def _on_open_dataroot(self):
        d = filedialog.askdirectory(title="Chọn thư mục chứa ảnh (data_root)", initialdir=str(self.data_root))
        if d:
            self.data_root = Path(d)
            self._thumb_cache.clear()
            self._load_dataset_images()
            if self.current_cluster_id is not None:
                self._render_cluster(self.current_cluster_id)

    def _on_open_current_folder(self):
        img_dir = self.data_root / "images"
        if img_dir.exists():
            os.startfile(str(img_dir))

    def _show_help(self):
        msg = (
            "HƯỚNG DẪN SỬ DỤNG BỘ XEM CỤM ẢNH:\n\n"
            "• Chọn Cụm: Click vào bất kỳ cụm nào ở cột bên trái để hiển thị toàn bộ ảnh của cụm đó.\n"
            "• Tìm kiếm: Nhập số Cụm ID (VD: 2022) hoặc tên file ảnh (VD: 2389).\n"
            "• Lọc cụm: Lọc theo Cụm đa ảnh (>=2 ảnh), Cụm lớn (>=5 ảnh) hoặc Tất cả.\n"
            "• Phím tắt bàn phím:\n"
            "    - [◄] / [►]: Chuyển nhanh cụm trước / cụm sau.\n"
            "    - [▲] / [▼]: Di chuyển trong danh sách cụm.\n"
            "• Bấm vào ảnh: Phóng to ảnh HD kèm Bounding box chi tiết.\n"
            "• Bật/Tắt Bounding Box: Dùng nút tích ở góc dưới bên trái."
        )
        messagebox.showinfo("Hướng dẫn", msg)


def main():
    root = tk.Tk()
    app = ClusterViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
