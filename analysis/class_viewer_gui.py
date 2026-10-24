"""Class Viewer GUI - Ứng dụng xem và kiểm tra ảnh theo từng lớp sau khi tiền xử lý (data/processed).

Giao diện Desktop hiện đại (Tkinter + PIL):
- Duyệt toàn bộ 52 lớp biển báo giao thông Việt Nam.
- Xem số lượng mẫu, phân bố chi tiết theo từng split: Train (Real vs Aug), Val, Test.
- Lọc theo split: Tất cả, Train Real, Train Aug, Val, Test.
- Lọc theo mức độ mẫu: Lớp ít mẫu (< 30), Lớp trung bình (30 - 150), Lớp nhiều mẫu (> 150).
- Hiển thị ảnh dạng lưới (Grid Thumbnail) với Bounding Box và tên tiếng Việt.
- Bấm vào ảnh để mở popup phóng to 640x640, xem danh sách box và crop cận cảnh biển báo.
- Phân trang mượt mà (Pagination) chống lag.

Cách chạy:
    python ./analysis/class_viewer_gui.py
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
from collections import defaultdict, Counter
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont


class ClassViewerApp:
    def __init__(self, root: tk.Tk, data_root: str = "data/processed", classes_csv: str = "configs/classes.csv"):
        self.root = root
        self.root.title("Class Viewer — Kiểm Định & Xem Ảnh Theo Lớp (Data Processed)")
        self.root.geometry("1360x860")
        self.root.minsize(1080, 700)

        self.data_root = Path(data_root).resolve()
        self.classes_csv_path = Path(classes_csv).resolve()
        self.cache_path = self.data_root / "class_index_cache.json"

        # Style & Theme
        self._setup_styles()

        # Data Structures
        self.classes: dict[int, str] = {}
        self.class_to_stems: dict[int, list[str]] = defaultdict(list)
        self.stem_to_data: dict[str, dict] = {}
        self.class_stats: dict[int, dict] = {}

        # Navigation state
        self.current_class_id: int | None = None
        self.current_page: int = 1
        self.items_per_page: int = 24
        self.filtered_stems: list[str] = []

        # UI Variables
        self.var_search = tk.StringVar(value="")
        self.var_tier_filter = tk.StringVar(value="Tất cả 52 lớp")
        self.var_sort = tk.StringVar(value="ID tăng dần")
        self.var_split_filter = tk.StringVar(value="Tất cả split")
        self.var_draw_boxes = tk.BooleanVar(value=True)
        self.var_only_target_class = tk.BooleanVar(value=True)
        self.var_thumb_size = tk.StringVar(value="Vừa (220px)")

        # Cache thumbnail ImageTk
        self._thumb_cache: dict[tuple[str, bool, bool, int], ImageTk.PhotoImage] = {}
        self._current_photo_refs: list[ImageTk.PhotoImage] = []

        # Build UI layout
        self._build_menu()
        self._build_layout()

        # Load data trực tiếp từ thư mục
        self._load_classes()
        self._scan_dataset()

        # Keyboard shortcuts
        self.root.bind("<Left>", lambda e: self._on_prev_page())
        self.root.bind("<Right>", lambda e: self._on_next_page())
        self.root.bind("<Up>", lambda e: self._on_arrow_class(-1))
        self.root.bind("<Down>", lambda e: self._on_arrow_class(1))
        self.root.bind("<F5>", lambda e: self._scan_dataset())

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
        self.c_accent_green = "#2A9D8F"
        self.c_accent_purple = "#7209B7"
        self.c_accent_orange = "#F4A261"
        self.c_text = "#2B2D42"
        self.c_subtext = "#6C757D"
        self.c_border = "#DEE2E6"

        self.root.configure(bg=self.c_bg)

        style.configure("Treeview", font=("Segoe UI", 9), rowheight=26)
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", self.c_accent_blue)], foreground=[("selected", "white")])

        style.configure("Primary.TButton", font=("Segoe UI", 9, "bold"), background=self.c_primary, foreground="white")
        style.configure("Accent.TButton", font=("Segoe UI", 9, "bold"), background=self.c_accent, foreground="white")

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tệp (File)", menu=file_menu)
        file_menu.add_command(label="Mở thư mục ảnh...", command=self._on_open_data_folder)
        file_menu.add_command(label="Làm mới dữ liệu (F5)...", command=lambda: self._reload_data(force_rebuild=True))
        file_menu.add_separator()
        file_menu.add_command(label="Thoát", command=self.root.quit)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Trợ giúp", menu=help_menu)
        help_menu.add_command(label="Hướng dẫn & Phím tắt", command=self._show_help)

    # ──────────────────────────────────────────────────────────────────────────
    # Layout Builder
    # ──────────────────────────────────────────────────────────────────────────
    def _build_layout(self):
        # 1. Top Header Bar
        self.top_bar = tk.Frame(self.root, bg=self.c_primary, height=46)
        self.top_bar.pack(side=tk.TOP, fill=tk.X)
        self.top_bar.pack_propagate(False)

        self.lbl_header = tk.Label(
            self.top_bar,
            text="🏷️ Class Viewer — Trực Quan Hóa Dữ Liệu Theo Lớp (Data Processed)",
            font=("Segoe UI", 12, "bold"),
            fg="white",
            bg=self.c_primary,
            padx=16,
        )
        self.lbl_header.pack(side=tk.LEFT, fill=tk.Y)

        self.lbl_top_stats = tk.Label(
            self.top_bar,
            text="Đang nạp dữ liệu...",
            font=("Segoe UI", 10),
            fg="#A8DADC",
            bg=self.c_primary,
            padx=16,
        )
        self.lbl_top_stats.pack(side=tk.RIGHT, fill=tk.Y)

        # 2. Main Paned Window
        self.main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bg=self.c_border, sashwidth=5)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        # ──────────────────────────────────────────────────────────────────────
        # LEFT SIDEBAR: Class List & Search
        # ──────────────────────────────────────────────────────────────────────
        self.left_frame = tk.Frame(self.main_paned, bg=self.c_bg, width=420, padx=10, pady=10)
        self.left_frame.pack_propagate(False)
        self.main_paned.add(self.left_frame)

        # Search Bar
        lbl_s = tk.Label(self.left_frame, text="🔍 Tìm Lớp (ID hoặc Tên tiếng Việt):", font=("Segoe UI", 9, "bold"), bg=self.c_bg, fg=self.c_text)
        lbl_s.pack(anchor=tk.W, pady=(0, 2))

        self.entry_search = ttk.Entry(self.left_frame, textvariable=self.var_search, font=("Segoe UI", 10))
        self.entry_search.pack(fill=tk.X, pady=(0, 6))
        self.entry_search.bind("<KeyRelease>", lambda e: self._apply_class_filter())

        # Filter & Sort controls
        fs_frame = tk.Frame(self.left_frame, bg=self.c_bg)
        fs_frame.pack(fill=tk.X, pady=(0, 6))

        tk.Label(fs_frame, text="Lọc:", font=("Segoe UI", 9), bg=self.c_bg, fg=self.c_subtext).pack(side=tk.LEFT, padx=(0, 4))
        self.cb_tier = ttk.Combobox(
            fs_frame,
            textvariable=self.var_tier_filter,
            values=["Tất cả 52 lớp", "Lớp ít mẫu (< 30)", "Lớp trung bình (30 - 150)", "Lớp nhiều mẫu (> 150)"],
            state="readonly",
            width=17,
            font=("Segoe UI", 9),
        )
        self.cb_tier.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.cb_tier.bind("<<ComboboxSelected>>", lambda e: self._apply_class_filter())

        self.cb_sort = ttk.Combobox(
            fs_frame,
            textvariable=self.var_sort,
            values=["ID tăng dần", "Tổng box giảm dần", "Số box tăng dần", "Tên A-Z"],
            state="readonly",
            width=15,
            font=("Segoe UI", 9),
        )
        self.cb_sort.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cb_sort.bind("<<ComboboxSelected>>", lambda e: self._apply_class_filter())

        # Treeview for 52 classes
        tree_container = tk.Frame(self.left_frame, bg=self.c_bg)
        tree_container.pack(fill=tk.BOTH, expand=True, pady=(2, 6))

        cols = ("id", "name", "total", "train", "val", "test")
        self.tree_classes = ttk.Treeview(tree_container, columns=cols, show="headings", selectmode="browse")
        self.tree_classes.heading("id", text="ID", anchor=tk.CENTER)
        self.tree_classes.heading("name", text="Tên biển báo", anchor=tk.W)
        self.tree_classes.heading("total", text="Tổng", anchor=tk.CENTER)
        self.tree_classes.heading("train", text="Train", anchor=tk.CENTER)
        self.tree_classes.heading("val", text="Val", anchor=tk.CENTER)
        self.tree_classes.heading("test", text="Test", anchor=tk.CENTER)

        self.tree_classes.column("id", width=36, anchor=tk.CENTER, stretch=False)
        self.tree_classes.column("name", width=175, anchor=tk.W)
        self.tree_classes.column("total", width=48, anchor=tk.CENTER, stretch=False)
        self.tree_classes.column("train", width=52, anchor=tk.CENTER, stretch=False)
        self.tree_classes.column("val", width=42, anchor=tk.CENTER, stretch=False)
        self.tree_classes.column("test", width=42, anchor=tk.CENTER, stretch=False)

        scroll_tree = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.tree_classes.yview)
        self.tree_classes.configure(yscrollcommand=scroll_tree.set)
        self.tree_classes.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_tree.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree_classes.bind("<<TreeviewSelect>>", self._on_class_select)

        # Bottom Left: Quick info
        self.lbl_class_quick_info = tk.Label(
            self.left_frame,
            text="Chọn một lớp trong bảng để xem danh sách ảnh.",
            font=("Segoe UI", 9, "italic"),
            fg=self.c_subtext,
            bg=self.c_bg,
            wraplength=400,
            justify=tk.LEFT,
        )
        self.lbl_class_quick_info.pack(fill=tk.X, pady=(2, 0))

        # ──────────────────────────────────────────────────────────────────────
        # RIGHT PANEL: Image Grid & Viewer
        # ──────────────────────────────────────────────────────────────────────
        self.right_frame = tk.Frame(self.main_paned, bg=self.c_bg, padx=12, pady=10)
        self.main_paned.add(self.right_frame)

        # Class Header Card
        self.card_header = tk.Frame(self.right_frame, bg=self.c_card, bd=1, relief=tk.SOLID, padx=14, pady=10)
        self.card_header.pack(fill=tk.X, pady=(0, 8))

        self.lbl_current_title = tk.Label(
            self.card_header,
            text="Vui lòng chọn lớp biển báo để duyệt ảnh",
            font=("Segoe UI", 13, "bold"),
            fg=self.c_primary,
            bg=self.c_card,
        )
        self.lbl_current_title.pack(anchor=tk.W)

        self.lbl_current_subtitle = tk.Label(
            self.card_header,
            text="Tập dữ liệu: data/processed",
            font=("Segoe UI", 9),
            fg=self.c_subtext,
            bg=self.c_card,
        )
        self.lbl_current_subtitle.pack(anchor=tk.W, pady=(2, 0))

        # Filter and View Controls Toolbar
        toolbar = tk.Frame(self.right_frame, bg=self.c_bg)
        toolbar.pack(fill=tk.X, pady=(0, 8))

        # Split filter
        tk.Label(toolbar, text="Split:", font=("Segoe UI", 9, "bold"), bg=self.c_bg, fg=self.c_text).pack(side=tk.LEFT, padx=(0, 4))
        self.cb_split = ttk.Combobox(
            toolbar,
            textvariable=self.var_split_filter,
            values=[
                "Tất cả split",
                "Train (Tất cả)",
                "Train (Ảnh gốc - Real)",
                "Train (Ảnh tăng cường - Aug)",
                "Val (Validation)",
                "Test",
            ],
            state="readonly",
            width=23,
            font=("Segoe UI", 9),
        )
        self.cb_split.pack(side=tk.LEFT, padx=(0, 12))
        self.cb_split.bind("<<ComboboxSelected>>", lambda e: self._on_split_filter_change())

        # Checkboxes
        chk_box = tk.Checkbutton(
            toolbar,
            text="Vẽ Bounding Box",
            variable=self.var_draw_boxes,
            command=self._render_current_page,
            font=("Segoe UI", 9, "bold"),
            bg=self.c_bg,
            fg=self.c_primary,
            activebackground=self.c_bg,
        )
        chk_box.pack(side=tk.LEFT, padx=(0, 8))

        chk_only = tk.Checkbutton(
            toolbar,
            text="Chỉ highlight lớp đang chọn",
            variable=self.var_only_target_class,
            command=self._render_current_page,
            font=("Segoe UI", 9),
            bg=self.c_bg,
            fg=self.c_text,
            activebackground=self.c_bg,
        )
        chk_only.pack(side=tk.LEFT, padx=(0, 12))

        # Thumb size
        tk.Label(toolbar, text="Cỡ ảnh:", font=("Segoe UI", 9), bg=self.c_bg, fg=self.c_subtext).pack(side=tk.LEFT, padx=(0, 4))
        self.cb_size = ttk.Combobox(
            toolbar,
            textvariable=self.var_thumb_size,
            values=["Nhỏ (170px)", "Vừa (220px)", "Lớn (280px)"],
            state="readonly",
            width=12,
            font=("Segoe UI", 9),
        )
        self.cb_size.pack(side=tk.LEFT, padx=(0, 10))
        self.cb_size.bind("<<ComboboxSelected>>", lambda e: self._render_current_page())

        # Pagination controls
        self.page_frame = tk.Frame(toolbar, bg=self.c_bg)
        self.page_frame.pack(side=tk.RIGHT)

        self.btn_prev = ttk.Button(self.page_frame, text="◄ Trước", command=self._on_prev_page, width=8)
        self.btn_prev.pack(side=tk.LEFT, padx=2)

        self.lbl_page_info = tk.Label(self.page_frame, text="Trang 0 / 0", font=("Segoe UI", 9, "bold"), bg=self.c_bg, fg=self.c_text, padx=6)
        self.lbl_page_info.pack(side=tk.LEFT)

        self.btn_next = ttk.Button(self.page_frame, text="Sau ►", command=self._on_next_page, width=8)
        self.btn_next.pack(side=tk.LEFT, padx=2)

        # Image Grid with Canvas & Scrollbar
        grid_container = tk.Frame(self.right_frame, bg=self.c_card, bd=1, relief=tk.SOLID)
        grid_container.pack(fill=tk.BOTH, expand=True)

        self.canvas_grid = tk.Canvas(grid_container, bg=self.c_card, highlightthickness=0)
        self.scroll_grid = ttk.Scrollbar(grid_container, orient=tk.VERTICAL, command=self.canvas_grid.yview)
        self.canvas_grid.configure(yscrollcommand=self.scroll_grid.set)

        self.scroll_grid.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_grid.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.inner_grid_frame = tk.Frame(self.canvas_grid, bg=self.c_card)
        self.canvas_window_id = self.canvas_grid.create_window((0, 0), window=self.inner_grid_frame, anchor="nw")

        self.inner_grid_frame.bind("<Configure>", lambda e: self.canvas_grid.configure(scrollregion=self.canvas_grid.bbox("all")))
        self.canvas_grid.bind("<Configure>", self._on_canvas_configure)
        self.canvas_grid.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_canvas_configure(self, event):
        self.canvas_grid.itemconfig(self.canvas_window_id, width=event.width)

    def _on_mousewheel(self, event):
        # Cuộn trang bằng con lăn chuột
        if self.canvas_grid.winfo_exists():
            self.canvas_grid.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ──────────────────────────────────────────────────────────────────────────
    # Data Loading & Indexing
    # ──────────────────────────────────────────────────────────────────────────
    def _load_classes(self):
        """Đọc configs/classes.csv."""
        if not self.classes_csv_path.exists():
            messagebox.showerror("Lỗi", f"Không tìm thấy file: {self.classes_csv_path}")
            return
        with open(self.classes_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = int(row["class_id"])
                self.classes[cid] = row["name_vi"]

    def _scan_dataset(self):
        """Quét trực tiếp data/processed bằng os.scandir siêu tốc (~0.4s).
        Tự động cập nhật 100% dữ liệu mới nhất mỗi khi mở app hoặc sau khi chạy preprocess.
        """
        self.lbl_top_stats.config(text="Đang nạp dữ liệu trực tiếp từ ổ đĩa...")
        self.root.update_idletasks()

        lbl_dir = self.data_root / "labels"
        if not lbl_dir.is_dir():
            messagebox.showwarning("Cảnh báo", f"Không tìm thấy thư mục nhãn: {lbl_dir}")
            return

        self._thumb_cache.clear()
        self.class_to_stems = defaultdict(list)
        self.stem_to_data = {}

        import time
        t0 = time.time()

        for sname in ("train", "val", "test"):
            s_lbl_dir = lbl_dir / sname
            if not s_lbl_dir.is_dir():
                continue

            with os.scandir(s_lbl_dir) as entries:
                for entry in entries:
                    if entry.name.endswith(".txt"):
                        stem = entry.name[:-4]
                        boxes = []
                        try:
                            with open(entry.path, "r", encoding="utf-8") as f:
                                for line in f:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    parts = line.split()
                                    if len(parts) >= 5:
                                        cid = int(parts[0])
                                        cx, cy, bw, bh = map(float, parts[1:5])
                                        boxes.append((cid, cx, cy, bw, bh))
                                        if stem not in self.class_to_stems[cid]:
                                            self.class_to_stems[cid].append(stem)
                        except Exception:
                            continue

                        img_path = self.data_root / "images" / sname / f"{stem}.jpg"
                        self.stem_to_data[stem] = {
                            "split": sname,
                            "is_aug": ("_aug" in stem),
                            "boxes": boxes,
                            "img_path": str(img_path.resolve()),
                        }

        self._compute_class_stats()
        self._populate_class_tree()

        elapsed = time.time() - t0
        total_boxes = sum(len(v) for v in self.class_to_stems.values())
        self.lbl_top_stats.config(
            text=f"52 Lớp | {len(self.stem_to_data):,} Ảnh ({total_boxes:,} boxes) — Quét trực tiếp ({elapsed:.2f}s)"
        )

        # Chọn lại class hiện tại hoặc class đầu tiên
        if self.classes:
            target_id = self.current_class_id if self.current_class_id in self.classes else sorted(self.classes.keys())[0]
            self._select_class(target_id)

    def _compute_class_stats(self):
        """Tính toán thống kê box theo từng split cho mỗi class."""
        self.class_stats = {}
        for cid in self.classes:
            stems = self.class_to_stems.get(cid, [])
            total_boxes = 0
            train_real = 0
            train_aug = 0
            val_boxes = 0
            test_boxes = 0

            for st in stems:
                meta = self.stem_to_data.get(st, {})
                split = meta.get("split", "")
                is_aug = meta.get("is_aug", False)
                boxes = meta.get("boxes", [])
                matching_boxes = [b for b in boxes if b[0] == cid]
                n_b = len(matching_boxes)
                total_boxes += n_b

                if split == "train":
                    if is_aug:
                        train_aug += n_b
                    else:
                        train_real += n_b
                elif split == "val":
                    val_boxes += n_b
                elif split == "test":
                    test_boxes += n_b

            self.class_stats[cid] = {
                "total": total_boxes,
                "train_real": train_real,
                "train_aug": train_aug,
                "train_total": train_real + train_aug,
                "val": val_boxes,
                "test": test_boxes,
            }

    # ──────────────────────────────────────────────────────────────────────────
    # Filtering & Tree Population
    # ──────────────────────────────────────────────────────────────────────────
    def _populate_class_tree(self):
        """Đưa danh sách lớp vào Treeview với bộ lọc và sắp xếp."""
        for item in self.tree_classes.get_children():
            self.tree_classes.delete(item)

        query = self.var_search.get().strip().lower()
        tier_filter = self.var_tier_filter.get()
        sort_mode = self.var_sort.get()

        visible_ids = []
        for cid, cname in self.classes.items():
            stats = self.class_stats.get(cid, {})
            tot = stats.get("total", 0)

            # 1. Search Query
            if query:
                match_id = str(cid) == query
                match_name = query in cname.lower()
                if not (match_id or match_name):
                    continue

            # 2. Tier Filter
            if tier_filter == "Lớp ít mẫu (< 30)":
                if tot >= 30:
                    continue
            elif tier_filter == "Lớp trung bình (30 - 150)":
                if tot < 30 or tot > 150:
                    continue
            elif tier_filter == "Lớp nhiều mẫu (> 150)":
                if tot <= 150:
                    continue

            visible_ids.append(cid)

        # 3. Sort
        if sort_mode == "ID tăng dần":
            visible_ids.sort()
        elif sort_mode == "Tổng box giảm dần":
            visible_ids.sort(key=lambda x: self.class_stats.get(x, {}).get("total", 0), reverse=True)
        elif sort_mode == "Số box tăng dần":
            visible_ids.sort(key=lambda x: self.class_stats.get(x, {}).get("total", 0))
        elif sort_mode == "Tên A-Z":
            visible_ids.sort(key=lambda x: self.classes.get(x, ""))

        # Insert to tree
        for cid in visible_ids:
            cname = self.classes.get(cid, f"Class {cid}")
            stats = self.class_stats.get(cid, {})
            tot = stats.get("total", 0)
            tr = f"{stats.get('train_real', 0)}+{stats.get('train_aug', 0)}" if stats.get('train_aug', 0) > 0 else f"{stats.get('train_real', 0)}"
            va = str(stats.get("val", 0))
            te = str(stats.get("test", 0))

            item_id = self.tree_classes.insert(
                "",
                tk.END,
                iid=str(cid),
                values=(cid, cname, tot, tr, va, te),
            )

        # Highlight current class if still visible
        if self.current_class_id is not None and str(self.current_class_id) in self.tree_classes.get_children():
            self.tree_classes.selection_set(str(self.current_class_id))
            self.tree_classes.see(str(self.current_class_id))

    def _apply_class_filter(self):
        self._populate_class_tree()

    def _on_class_select(self, event):
        sel = self.tree_classes.selection()
        if not sel:
            return
        cid = int(sel[0])
        self._select_class(cid)

    def _select_class(self, cid: int):
        self.current_class_id = cid
        self.current_page = 1
        cname = self.classes.get(cid, f"Lớp {cid}")
        stats = self.class_stats.get(cid, {})

        tot = stats.get("total", 0)
        tr_real = stats.get("train_real", 0)
        tr_aug = stats.get("train_aug", 0)
        val_cnt = stats.get("val", 0)
        test_cnt = stats.get("test", 0)

        self.lbl_current_title.config(text=f"Lớp [{cid:02d}]: {cname}")
        self.lbl_current_subtitle.config(
            text=f"Tổng số box: {tot:,}  |  Train: {tr_real} Real + {tr_aug} Aug (Tổng {tr_real+tr_aug})  |  Val: {val_cnt}  |  Test: {test_cnt}"
        )
        self.lbl_class_quick_info.config(
            text=f"Đang xem Lớp [{cid}] {cname}: {tot} box ({tr_real} ảnh gốc + {tr_aug} ảnh augmented)."
        )

        self._filter_class_stems()
        self._render_current_page()

    def _on_split_filter_change(self):
        self.current_page = 1
        self._filter_class_stems()
        self._render_current_page()

    def _filter_class_stems(self):
        """Lọc danh sách ảnh theo split đang chọn."""
        if self.current_class_id is None:
            self.filtered_stems = []
            return

        all_stems = self.class_to_stems.get(self.current_class_id, [])
        split_opt = self.var_split_filter.get()

        res = []
        for st in all_stems:
            meta = self.stem_to_data.get(st, {})
            sp = meta.get("split", "")
            is_aug = meta.get("is_aug", False)

            if split_opt == "Tất cả split":
                res.append(st)
            elif split_opt == "Train (Tất cả)" and sp == "train":
                res.append(st)
            elif split_opt == "Train (Ảnh gốc - Real)" and sp == "train" and not is_aug:
                res.append(st)
            elif split_opt == "Train (Ảnh tăng cường - Aug)" and sp == "train" and is_aug:
                res.append(st)
            elif split_opt == "Val (Validation)" and sp == "val":
                res.append(st)
            elif split_opt == "Test" and sp == "test":
                res.append(st)

        # Sắp xếp: ảnh thật lên trước, ảnh aug theo sau
        res.sort(key=lambda s: (self.stem_to_data.get(s, {}).get("is_aug", False), s))
        self.filtered_stems = res

    # ──────────────────────────────────────────────────────────────────────────
    # Rendering Image Grid
    # ──────────────────────────────────────────────────────────────────────────
    def _render_current_page(self):
        """Vẽ lưới ảnh của trang hiện tại."""
        for widget in self.inner_grid_frame.winfo_children():
            widget.destroy()

        self._current_photo_refs.clear()

        total_items = len(self.filtered_stems)
        if total_items == 0:
            self.lbl_page_info.config(text="Trang 0 / 0 (0 ảnh)")
            no_lbl = tk.Label(
                self.inner_grid_frame,
                text="Không có ảnh nào phù hợp với bộ lọc hiện tại.",
                font=("Segoe UI", 11, "italic"),
                fg=self.c_subtext,
                bg=self.c_card,
                pady=40,
            )
            no_lbl.pack()
            return

        max_page = (total_items + self.items_per_page - 1) // self.items_per_page
        self.current_page = max(1, min(self.current_page, max_page))
        self.lbl_page_info.config(text=f"Trang {self.current_page} / {max_page} ({total_items} ảnh)")

        start_idx = (self.current_page - 1) * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, total_items)
        page_stems = self.filtered_stems[start_idx:end_idx]

        # Determine thumb size px
        size_str = self.var_thumb_size.get()
        if "Nhỏ" in size_str:
            thumb_px = 170
        elif "Lớn" in size_str:
            thumb_px = 280
        else:
            thumb_px = 220

        # Calculate number of columns based on width
        canvas_w = self.canvas_grid.winfo_width()
        col_w = thumb_px + 24
        cols = max(1, canvas_w // col_w)

        draw_boxes = self.var_draw_boxes.get()
        only_target = self.var_only_target_class.get()

        for idx, stem in enumerate(page_stems):
            r = idx // cols
            c = idx % cols
            meta = self.stem_to_data.get(stem, {})
            sp = meta.get("split", "train")
            is_aug = meta.get("is_aug", False)
            boxes = meta.get("boxes", [])
            img_p = Path(meta.get("img_path", ""))

            # Card container
            card = tk.Frame(self.inner_grid_frame, bg="white", bd=1, relief=tk.SOLID, padx=6, pady=6)
            card.grid(row=r, column=c, padx=8, pady=8, sticky="nsew")

            # Thumbnail Image
            thumb_img = self._get_rendered_thumbnail(img_p, boxes, thumb_px, draw_boxes, only_target)
            if thumb_img:
                self._current_photo_refs.append(thumb_img)
                lbl_img = tk.Label(card, image=thumb_img, bg="white", cursor="hand2")
                lbl_img.pack()
                lbl_img.bind("<Button-1>", lambda e, s=stem: self._open_detail_popup(s))
                lbl_img.bind("<Double-Button-1>", lambda e, s=stem: self._open_detail_popup(s))
            else:
                lbl_no = tk.Label(card, text="[Ảnh không tìm thấy]", width=20, height=8, bg="#ECEFF1", fg="#90A4AE")
                lbl_no.pack()

            # Split Badge + Filename
            meta_row = tk.Frame(card, bg="white")
            meta_row.pack(fill=tk.X, pady=(4, 0))

            if sp == "train":
                badge_bg = self.c_accent_purple if is_aug else self.c_accent_green
                badge_text = "TRAIN (AUG)" if is_aug else "TRAIN (REAL)"
            elif sp == "val":
                badge_bg = self.c_accent_orange
                badge_text = "VAL"
            else:
                badge_bg = self.c_accent_blue
                badge_text = "TEST"

            badge = tk.Label(
                meta_row,
                text=badge_text,
                font=("Segoe UI", 7, "bold"),
                fg="white",
                bg=badge_bg,
                padx=4,
                pady=1,
            )
            badge.pack(side=tk.LEFT)

            n_target_b = sum(1 for b in boxes if b[0] == self.current_class_id)
            lbl_bcnt = tk.Label(
                meta_row,
                text=f"{n_target_b} box",
                font=("Segoe UI", 8, "bold"),
                fg=self.c_accent,
                bg="white",
            )
            lbl_bcnt.pack(side=tk.RIGHT)

            # Stem name label
            lbl_name = tk.Label(
                card,
                text=f"{stem}.jpg",
                font=("Segoe UI", 8),
                fg=self.c_text,
                bg="white",
                anchor=tk.W,
            )
            lbl_name.pack(fill=tk.X, pady=(2, 0))

        # Reset scroll to top on page change
        self.canvas_grid.yview_moveto(0.0)

    def _get_rendered_thumbnail(
        self,
        img_p: Path,
        boxes: list,
        thumb_size: int,
        draw_boxes: bool,
        only_target: bool,
    ) -> ImageTk.PhotoImage | None:
        """Tạo thumbnail có vẽ bounding box và lưu cache."""
        cache_key = (str(img_p), draw_boxes, only_target, thumb_size)
        if cache_key in self._thumb_cache:
            return self._thumb_cache[cache_key]

        if not img_p.exists():
            stem = img_p.stem
            found = False
            for sname in ("train", "val", "test"):
                alt = self.data_root / "images" / sname / f"{stem}.jpg"
                if alt.exists():
                    img_p = alt
                    found = True
                    break
            if not found:
                return None

        try:
            im = Image.open(img_p).convert("RGB")
            w, h = im.size

            if draw_boxes and boxes:
                draw = ImageDraw.Draw(im)
                try:
                    font = ImageFont.truetype("arial.ttf", 13)
                except Exception:
                    font = ImageFont.load_default()

                for b in boxes:
                    cid, cx, cy, bw, bh = b
                    is_target = (cid == self.current_class_id)
                    if only_target and not is_target:
                        continue

                    bx1 = (cx - bw / 2.0) * w
                    by1 = (cy - bh / 2.0) * h
                    bx2 = (cx + bw / 2.0) * w
                    by2 = (cy + bh / 2.0) * h

                    color = "#E63946" if is_target else "#457B9D"
                    # Vẽ viền box dày 3px
                    draw.rectangle([bx1, by1, bx2, by2], outline=color, width=3)

                    # Tag nhãn
                    cname = self.classes.get(cid, f"Lớp {cid}")[:14]
                    tag_text = f"[{cid}] {cname}"
                    draw.rectangle([bx1, max(0, by1 - 16), bx1 + len(tag_text) * 7.5, max(16, by1)], fill=color)
                    draw.text((bx1 + 2, max(0, by1 - 16)), tag_text, fill="white", font=font)

            # Resize to thumbnail square
            im.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(im)
            if len(self._thumb_cache) < 600:
                self._thumb_cache[cache_key] = photo
            return photo
        except Exception as e:
            print(f"[ClassViewer] Lỗi nạp ảnh {img_p}: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Detail Popup (Kích thước gốc & Box Inspector)
    # ──────────────────────────────────────────────────────────────────────────
    def _open_detail_popup(self, stem: str):
        """Mở popup xem ảnh phóng to 640x640 và chi tiết từng bounding box."""
        meta = self.stem_to_data.get(stem, {})
        img_p = Path(meta.get("img_path", ""))
        boxes = meta.get("boxes", [])
        sp = meta.get("split", "")
        is_aug = meta.get("is_aug", False)

        if not img_p.exists():
            for sname in ("train", "val", "test"):
                alt = self.data_root / "images" / sname / f"{stem}.jpg"
                if alt.exists():
                    img_p = alt
                    break
        if not img_p.exists():
            messagebox.showwarning("Cảnh báo", f"Không tìm thấy file: {img_p}")
            return

        win = tk.Toplevel(self.root)
        win.title(f"Chi tiết ảnh: {stem}.jpg — [{sp.upper()}]")
        win.geometry("980x740")
        win.minsize(800, 600)
        win.configure(bg=self.c_bg)

        # Header of Popup
        top_p = tk.Frame(win, bg=self.c_primary, height=42, padx=14)
        top_p.pack(fill=tk.X)
        top_p.pack_propagate(False)

        badge_txt = f"[{sp.upper()}] - {'ẢNH TĂNG CƯỜNG (AUG)' if is_aug else 'ẢNH GỐC (REAL)'}"
        tk.Label(top_p, text=f"📷 {stem}.jpg  |  {badge_txt}", font=("Segoe UI", 11, "bold"), fg="white", bg=self.c_primary).pack(side=tk.LEFT)

        btn_open_exp = ttk.Button(top_p, text="Mở trong Explorer", command=lambda: self._reveal_in_explorer(img_p))
        btn_open_exp.pack(side=tk.RIGHT, pady=6)

        # Content paned
        pane = tk.PanedWindow(win, orient=tk.HORIZONTAL, bg=self.c_border, sashwidth=4)
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Left: Large Image
        left_box = tk.Frame(pane, bg="white", bd=1, relief=tk.SOLID)
        pane.add(left_box, width=640)

        try:
            im = Image.open(img_p).convert("RGB")
            w, h = im.size
            draw = ImageDraw.Draw(im)
            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except Exception:
                font = ImageFont.load_default()

            for b in boxes:
                cid, cx, cy, bw, bh = b
                is_target = (cid == self.current_class_id)
                bx1 = (cx - bw / 2.0) * w
                by1 = (cy - bh / 2.0) * h
                bx2 = (cx + bw / 2.0) * w
                by2 = (cy + bh / 2.0) * h

                color = "#E63946" if is_target else "#457B9D"
                draw.rectangle([bx1, by1, bx2, by2], outline=color, width=3)
                cname = self.classes.get(cid, f"Lớp {cid}")
                tag_t = f"[{cid}] {cname}"
                draw.rectangle([bx1, max(0, by1 - 20), bx1 + len(tag_t) * 9, max(20, by1)], fill=color)
                draw.text((bx1 + 3, max(0, by1 - 18)), tag_t, fill="white", font=font)

            photo_large = ImageTk.PhotoImage(im)
            lbl_large = tk.Label(left_box, image=photo_large, bg="white")
            lbl_large.image = photo_large
            lbl_large.pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            tk.Label(left_box, text=f"Lỗi hiển thị: {e}").pack(pady=40)

        # Right: Box List & Info
        right_box = tk.Frame(pane, bg=self.c_bg, padx=8, pady=8)
        pane.add(right_box, width=320)

        tk.Label(right_box, text=f"Danh sách {len(boxes)} Bounding Box trong ảnh:", font=("Segoe UI", 10, "bold"), bg=self.c_bg, fg=self.c_primary).pack(anchor=tk.W, pady=(0, 6))

        for idx, b in enumerate(boxes, start=1):
            cid, cx, cy, bw, bh = b
            is_target = (cid == self.current_class_id)
            cname = self.classes.get(cid, f"Lớp {cid}")

            card_b = tk.Frame(right_box, bg="white" if not is_target else "#FFF0F0", bd=1, relief=tk.SOLID, padx=8, pady=6)
            card_b.pack(fill=tk.X, pady=4)

            tk.Label(
                card_b,
                text=f"Box #{idx}: [{cid}] {cname}",
                font=("Segoe UI", 9, "bold"),
                fg=self.c_accent if is_target else self.c_accent_blue,
                bg="white" if not is_target else "#FFF0F0",
            ).pack(anchor=tk.W)

            coords_t = f"Tâm: (cx={cx:.4f}, cy={cy:.4f})\nCỡ : {bw*640:.1f}×{bh*640:.1f} px  (w={bw:.3f}, h={bh:.3f})"
            tk.Label(
                card_b,
                text=coords_t,
                font=("Consolas", 8),
                fg=self.c_text,
                bg="white" if not is_target else "#FFF0F0",
                justify=tk.LEFT,
            ).pack(anchor=tk.W, pady=(2, 0))

        win.bind("<Escape>", lambda e: win.destroy())

    def _reveal_in_explorer(self, file_path: Path):
        """Mở vị trí file trong Windows Explorer."""
        if file_path.exists():
            try:
                subprocess.run(["explorer", f"/select,{str(file_path)}"])
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể mở Explorer: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Pagination & Navigation
    # ──────────────────────────────────────────────────────────────────────────
    def _on_prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self._render_current_page()

    def _on_next_page(self):
        total_items = len(self.filtered_stems)
        max_page = (total_items + self.items_per_page - 1) // self.items_per_page
        if self.current_page < max_page:
            self.current_page += 1
            self._render_current_page()

    def _on_arrow_class(self, direction: int):
        """Chuyển class tiếp theo hoặc trước đó bằng phím mũi tên Lên/Xuống."""
        items = list(self.tree_classes.get_children())
        if not items:
            return
        curr = self.tree_classes.selection()
        if not curr:
            next_idx = 0
        else:
            try:
                curr_idx = items.index(curr[0])
                next_idx = max(0, min(len(items) - 1, curr_idx + direction))
            except ValueError:
                next_idx = 0

        target_id = items[next_idx]
        self.tree_classes.selection_set(target_id)
        self.tree_classes.see(target_id)
        self._select_class(int(target_id))

    def _reload_data(self, force_rebuild: bool = True):
        self._thumb_cache.clear()
        self._load_or_build_index(force_rebuild=force_rebuild)

    def _on_open_data_folder(self):
        d = filedialog.askdirectory(initialdir=str(self.data_root), title="Chọn thư mục dữ liệu đã tiền xử lý")
        if d:
            self.data_root = Path(d).resolve()
            self.cache_path = self.data_root / "class_index_cache.json"
            self._reload_data(force_rebuild=True)

    def _show_help(self):
        msg = (
            "🏷️ Class Viewer — Hướng dẫn sử dụng:\n\n"
            "• Chọn Lớp: Click vào bất kỳ lớp nào ở danh sách bên trái để tải ảnh.\n"
            "• Tìm kiếm: Gõ Class ID hoặc tên tiếng Việt vào ô tìm kiếm.\n"
            "• Lọc Split: Chọn hiển thị toàn bộ, chỉ ảnh Real, chỉ ảnh Aug, Val hoặc Test.\n"
            "• Phóng to chi tiết: Click hoặc Double click vào bất kỳ ảnh nào để xem 640x640 và toạ độ box.\n"
            "• Phím tắt:\n"
            "   - [▲] / [▼]: Chuyển sang lớp trước / lớp tiếp theo.\n"
            "   - [◄] / [►]: Lật trang trước / trang sau.\n"
            "   - [F5]: Nạp lại toàn bộ dữ liệu.\n"
            "   - [Esc]: Đóng popup chi tiết ảnh.\n"
        )
        messagebox.showinfo("Hướng dẫn & Phím tắt", msg)


def main():
    root = tk.Tk()
    app = ClassViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
