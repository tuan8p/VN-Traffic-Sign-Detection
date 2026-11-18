# Nhận Diện Biển Báo Giao Thông Việt Nam (VN-Traffic-Sign-Detection)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics-00ffff.svg)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Dự án môn học **Xử lý ảnh & Thị giác máy tính (IPCV)**: Nghiên cứu, xây dựng và so sánh thực nghiệm các giải thuật phát hiện vật thể (*Object Detection*) trên tập dữ liệu **52 lớp biển báo giao thông Việt Nam**. Dự án triển khai đối sánh giữa hai trường phái kiến trúc: **Faster R-CNN (ResNet-50 FPN)** và **YOLO26m**, kết hợp thuật toán **Phân tích hình học thị giác máy tính (Mục 3.2)** và xây dựng **Ứng dụng Web tương tác Streamlit (Mục 3.4)**.

---

## Mục Lục

1. [Tổng Quan Dự Án](#1-tổng-quan-dự-án)
2. [Cấu Trúc Repository](#2-cấu-trúc-repository)
3. [Quy Trình Tiền Xử Lý & Huấn Luyện](#3-quy-trình-tiền-xử-lý--huấn-luyện)
4. [Phân Tích Hình Học & Cấu Trúc Cảnh (Mục 3.2)](#4-phân-tích-hình-học--cấu-trúc-cảnh-mục-32)
5. [Ứng Dụng Web Demo Tương Tác (Mục 3.4)](#5-ứng-dụng-web-demo-tương-tác-mục-34)
6. [Hướng Dẫn Cài Đặt & Chạy Nhanh](#6-hướng-dẫn-cài-đặt--chạy-nhanh)
7. [Dòng Lệnh Điều Khiển (CLI Reference)](#7-dòng-lệnh-điều-khiển-cli-reference)
8. [Quy Chuẩn Đầu Ra (Artifacts)](#8-quy-chuẩn-đầu-ra-artifacts)

---

## 1. Tổng Quan Dự Án

Hệ thống giải quyết bài toán phát hiện và nhận diện 52 loại biển báo đường bộ Việt Nam trong các điều kiện môi trường giao thông phức tạp (thời tiết, ánh sáng, góc chụp nghiêng, che khuất một phần):

- **Two-stage Detector — Faster R-CNN (ResNet-50 + FPN)**:
  - Tận dụng mạng trích xuất đặc trưng đa tầng Feature Pyramid Network (FPN) kết hợp Region Proposal Network (RPN).
  - Tích hợp **Custom MLP Predictor Head** ($1024 \to 512 \to \text{ReLU} \to \text{Dropout} \to \text{Output}$) giúp giảm thiểu hiện tượng quá khớp trên các lớp biển báo có số lượng mẫu ít.
  - Tối ưu hóa hàm mất mát đa nhiệm gồm Cross-Entropy và Smooth L1.
- **One-stage Detector — YOLO26m**:
  - Tối ưu hóa trên nền tảng Ultralytics YOLO, cân bằng giữa tốc độ xử lý thời gian thực (*real-time latency*) và độ chính xác cao.
  - Tinh chỉnh siêu tham số với bộ tối ưu hóa AdamW, Cosine Annealing learning rate scheduler.
- **Phân tích hình học & cấu trúc cảnh (Mục 3.2)**:
  - Trích xuất cấu trúc biên và đường thẳng toàn cảnh bằng toán tử Canny và biến đổi Hough xác suất (`HoughLinesP`).
  - Ước lượng ma trận biến đổi phối cảnh 2D Homography ($\mathbf{H}_{3 \times 3}$) nhằm nắn thẳng (*Perspective Rectification*) các biển báo bị nghiêng góc về góc nhìn trực diện chính diện chuẩn (*Frontal View*).
- **Giao diện Web Demo trực quan (Mục 3.4)**:
  - Trực quan hóa suy luận song song giữa 2 mô hình trên cùng ảnh đầu vào.
  - Tự động nhận diện checkpoint, bộ nhớ đệm mô hình trong RAM, hỗ trợ upload ảnh an toàn và tương tác các tham số hình học thời gian thực.

---

## 2. Cấu Trúc Repository

```text
VN-Traffic-Sign-Detection/
├── analysis/                 # Mô-đun phân tích khám phá dữ liệu (EDA)
│   ├── figures/              # Biểu đồ phân tích dữ liệu gốc và sau tiền xử lý
│   ├── class_viewer_gui.py   # Ứng dụng duyệt trực quan từng lớp biển báo
│   ├── cluster_viewer_gui.py # Ứng dụng duyệt cụm ảnh tương đồng (dHash)
│   └── eda.py                # Pipeline phân tích thống kê tự động
├── configs/                  # Hệ thống cấu hình tập trung (.yaml, .csv)
│   ├── classes.csv           # Danh mục 52 lớp biển báo (mã hiệu, tên tiếng Việt)
│   ├── shared.yaml           # Cấu hình chung cho cả 2 pipeline (batch size, seed, metrics)
│   ├── pipelines/            # Cấu hình chi tiết cho Faster R-CNN và YOLO26m
│   └── runtime/              # Cấu hình môi trường (local, Kaggle multi-GPU)
├── data/                     # Thư mục chứa dữ liệu thô và dữ liệu sau tiền xử lý
├── demo/                     # Ứng dụng Web Demo Streamlit & Phân tích hình học
│   ├── app.py                # Giao diện người dùng Streamlit chính
│   ├── geometry.py           # Thuật toán Canny, HoughLinesP và 2D Homography
│   ├── inference.py          # Trình nạp mô hình, suy luận và vẽ bounding box
│   └── samples/              # Bộ ảnh mẫu thực tế phục vụ thử nghiệm nhanh
├── tools/                    # Các công cụ dòng lệnh CLI
│   ├── run_preprocess.py     # Script tiền xử lý dữ liệu offline
│   ├── run_train.py          # Script điều phối huấn luyện (hỗ trợ single & multi-GPU DDP)
│   ├── run_eval.py           # Script đánh giá độc lập checkpoint trên tập test
│   └── pack_outputs.py       # Script đóng gói kết quả thực nghiệm (.zip)
├── vn_tsd/                   # Gói mã nguồn cốt lõi (Core Library)
│   ├── data/                 # Xử lý dữ liệu offline và online augmentation
│   ├── eval/                 # Bộ đánh giá chỉ số COCOeval chuẩn hóa
│   ├── pipelines/            # Triển khai chi tiết cho Faster R-CNN và YOLO
│   └── runtime/              # Quản lý logging, W&B tracking và xuất artifact
├── pyproject.toml            # Cấu hình dependencies và đóng gói dự án
└── README.md                 # Tài liệu tổng quan dự án
```

---

## 3. Quy Trình Tiền Xử Lý & Huấn Luyện

### 3.1. Tiền xử lý dữ liệu Offline (Shared Cache)
Để bảo đảm tính khách quan và công bằng khi so sánh thực nghiệm, cả hai mô hình sử dụng chung tập dữ liệu sau bước tiền xử lý offline:
- **Chuẩn hóa kích thước**: Đưa ảnh về độ phân giải $640 \times 640$ px bằng kỹ thuật Letterbox (bảo toàn tỷ lệ khung hình, bù đệm $\text{pad} = 114$).
- **Không sử dụng phép lật ngang (No Horizontal Flip)**: Nhằm bảo toàn ngữ nghĩa của các biển báo có tính đối xứng gương (như rẽ trái / rẽ phải).
- **Tăng cường lớp thiểu số (Minority Augmentation)**: Áp dụng các phép biến đổi hình học và màu sắc thích ứng cho các lớp biển báo hiếm gặp.
- **Phân chia tập dữ liệu**: Phân chia theo tỷ lệ 70% Train, 15% Validation, 15% Test; sử dụng thuật toán sai phân băm ảnh (dHash) để ngăn ngừa hiện tượng rò rỉ dữ liệu giữa các tập.

### 3.2. Quy chuẩn huấn luyện & Đánh giá
- **Phần cứng**: Huấn luyện phân tán trên Kaggle 2× NVIDIA T4 GPU với `torchrun` (Distributed Data Parallel — DDP, Automatic Mixed Precision — AMP).
- **Theo dõi thực nghiệm**: Đồng bộ toàn bộ loss, learning rate và metrics thời gian thực qua Weights & Biases (W&B).
- **Quy tắc dừng sớm (Early Stopping)**: Giám sát chỉ số `mAP50-95` trên tập validation với patience = 10 epochs.
- **Đánh giá đa seed (Multi-seed Protocol)**: Chốt cấu hình tối ưu với seed 42, sau đó lặp lại thực nghiệm trên 5 seed ngẫu nhiên `[42, 7, 21, 123, 2026]` để tính giá trị trung bình và độ lệch chuẩn ($\mu \pm \sigma$).
- **Bộ chỉ số tiêu chuẩn COCO**:
  - $\text{mAP}_{50-95}$: Chỉ số chất lượng chính, tính trung bình qua các ngưỡng IoU từ 0.50 đến 0.95.
  - $\text{mAP}_{50}$, $\text{mAP}_{75}$: Độ chính xác tại các ngưỡng IoU cụ thể.
  - $\text{AR}_{100}$: Độ thu hồi trung bình (Average Recall) với tối đa 100 detections mỗi ảnh.

---

## 4. Phân Tích Hình Học & Cấu Trúc Cảnh (Mục 3.2)

Hệ thống kết hợp các phương pháp xử lý ảnh kinh điển để bổ trợ cho mô hình học sâu:

### 4.1. Trích xuất biên và đường thẳng cấu trúc toàn cảnh (3.2.A)
1. Ảnh được chuyển sang thang độ xám (*Grayscale*) và lọc nhiễu qua bộ lọc Gauss (*Gaussian Blur*).
2. Phát hiện các điểm biên có độ biến thiên gradient lớn bằng thuật toán **Canny Edge Detection** với hai ngưỡng $T_{low}$ và $T_{high}$.
3. Áp dụng **Probabilistic Hough Transform** (`cv2.HoughLinesP`) trên bản đồ biên để tìm các đoạn thẳng dài tương ứng với lề đường, vạch kẻ đường, cột biển báo hoặc viền công trình.
4. Kết quả cung cấp các lớp phủ trực quan (*Edges, Lines, Combined*) và chỉ số thống kê mật độ biên.

### 4.2. Nắn thẳng phối cảnh 2D Homography cho biển báo (3.2.B)
Khi xe di chuyển, biển báo thường được ghi hình ở các góc nghiêng chéo khiến hình dạng bị biến dạng phối cảnh. Giải thuật nắn thẳng hoạt động như sau:
1. Trích xuất vùng ROI chứa biển báo phát hiện được với tỷ lệ padding mở rộng an toàn.
2. Xác định 4 đỉnh góc của biển báo dựa trên phân tích contour (đa giác xấp xỉ `cv2.approxPolyDP` hoặc khung chữ nhật xoay tối thiểu `cv2.minAreaRect`).
3. Ước lượng ma trận biến đổi phối cảnh 2D Homography $\mathbf{H} \in \mathbb{R}^{3 \times 3}$ bằng thuật toán Direct Linear Transformation (DLT) từ 4 cặp điểm nguồn $\mathbf{x}_i$ và đích $\mathbf{x}'_i$:

$$\mathbf{x}'_i \sim \mathbf{H} \mathbf{x}_i \iff \begin{bmatrix} x'_i \\ y'_i \\ 1 \end{bmatrix} \sim \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}$$

4. Áp dụng phép chiếu ngược phối cảnh (`cv2.warpPerspective`) để tái tạo ảnh biển báo về mặt phẳng chính diện chuẩn kích thước $240 \times 240$ px, giúp việc nhận diện ký hiệu bên trong biển báo trở nên sắc nét và chính xác.

---

## 5. Ứng Dụng Web Demo Tương Tác (Mục 3.4)

Ứng dụng web được phát triển bằng Streamlit cung cấp giao diện trực quan phục vụ demo và đánh giá:

- **Nguồn ảnh linh hoạt**: Hỗ trợ tải tệp ảnh từ máy tính (kéo thả, nạp file `JPG`, `PNG`, `WebP`, `JFIF`), tự động chuẩn hóa EXIF và nền trong suốt; hoặc chọn ảnh có sẵn từ danh mục mẫu.
- **Tự động nhận diện Checkpoint**: Quét tự động thư mục `outputs/` để liên kết trọng số tốt nhất của Faster R-CNN và YOLO26m (kèm chế độ giả lập thông minh khi chưa có checkpoint).
- **Bộ nhớ đệm thông minh**: Giữ mô hình trong RAM sau lần tải đầu tiên, giúp thời gian phản hồi suy luận cực nhanh.
- **Bảng điều khiển tham số**: Tùy chỉnh thanh trượt ngưỡng tin cậy (Confidence Threshold), lựa chọn phần cứng tính toán (CPU / CUDA GPU), tinh chỉnh ngưỡng Canny và Hough trực tiếp.
- **Giao diện kết quả chuyên sâu**: Xem trực tiếp ảnh phân tích, bảng ma trận Homography $\mathbf{H}$ dạng bảng số thực và công thức toán học minh họa.

---

## 6. Hướng Dẫn Cài Đặt & Chạy Nhanh

### Yêu cầu môi trường
- Hệ điều hành: Windows, Linux hoặc macOS.
- Python: Phiên bản **3.10** đến **3.12** (Khuyến nghị **Python 3.12** để đảm bảo tính tương thích tối ưu của PyTorch).

### Các bước cài đặt

```bash
# 1. Khởi tạo môi trường ảo
python -m venv .venv

# 2. Kích hoạt môi trường:
# Trên Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Trên Linux/macOS:
source .venv/bin/activate

# 3. Nâng cấp pip và cài đặt gói cơ bản
pip install -U pip setuptools wheel
pip install -e ".[dev]"

# 4. Cài đặt các thư viện Deep Learning (PyTorch, TorchVision, Ultralytics, W&B)
pip install -e ".[detect]"

# 5. Cài đặt thư viện giao diện Demo Streamlit
pip install -e ".[demo]"

# 6. Khởi tạo tệp môi trường (nếu dùng W&B tracking)
copy .env.example .env
```

---

## 7. Dòng Lệnh Điều Khiển (CLI Reference)

Mọi thao tác trong dự án đều có thể thực hiện thông qua giao diện dòng lệnh:

```bash
# 1. Phân tích khám phá dữ liệu (EDA)
python -m analysis.eda --data-root <đường_dẫn_thư_mục_ảnh_gốc>

# 2. Chạy tiền xử lý dữ liệu offline (tạo cache letterbox 640x640)
python -m tools.run_preprocess --data-root <đường_dẫn_thư_mục_ảnh_gốc>

# 3. Huấn luyện mô hình Faster R-CNN (Local 1 thiết bị)
python -m tools.run_train --pipeline faster_rcnn --runtime configs/runtime/local.yaml --run-name frcnn_local_exp

# 4. Huấn luyện mô hình YOLO26m (Local 1 thiết bị)
python -m tools.run_train --pipeline yolo26m --runtime configs/runtime/local.yaml --run-name yolo_local_exp

# 5. Huấn luyện phân tán đa GPU trên Kaggle (2x T4 GPU DDP)
torchrun --nproc_per_node=2 -m tools.run_train --pipeline faster_rcnn --runtime configs/runtime/kaggle.yaml --run-name frcnn_s42_final
torchrun --nproc_per_node=2 -m tools.run_train --pipeline yolo26m --runtime configs/runtime/kaggle.yaml --run-name yolo_s42_final

# 6. Đánh giá độc lập checkpoint trên tập test
python -m tools.run_eval --pipeline faster_rcnn --run-dir outputs/faster_rcnn/<run_id>
python -m tools.run_eval --pipeline yolo26m --run-dir outputs/yolo26m/<run_id>

# 7. Khởi chạy ứng dụng Web Demo Streamlit
streamlit run demo/app.py
```

---

## 8. Quy Chuẩn Đầu Ra (Artifacts)

Mỗi phiên huấn luyện sau khi hoàn tất sẽ tự động đóng gói toàn diện kết quả vào tệp nén `run.zip` lưu trữ tại thư mục tương ứng:

```text
outputs/<pipeline>/<run_id>/
├── resolved_config.yaml         # Toàn bộ cấu hình siêu tham số hiệu dụng của lượt chạy
├── metrics.json                 # Kết quả đánh giá cuối cùng trên tập test (mAP, AR)
├── history.json                 # Lịch sử biến thiên của train loss, val loss, lr theo từng epoch
├── wandb_run.json               # Định danh và liên kết trực tiếp tới phiên theo dõi trên W&B
├── checkpoints/
│   └── checkpoint_best.pt       # Trọng số mô hình tốt nhất theo val mAP50-95
├── figures/                     # Đồ thị đường cong hội tụ loss và biểu đồ PR
└── run.zip                      # Tệp nén tổng hợp toàn bộ các artifact trên
```

---

## 9. Bản Quyền & Tham Khảo (License & References)

Dự án được phân phối dưới giấy phép **MIT License**. Chi tiết xem tại [LICENSE](LICENSE).

### Tài liệu tham khảo:
- Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. NeurIPS.
- Lin, T. Y., et al. (2017). *Feature Pyramid Networks for Object Detection*. CVPR.
- Ultralytics. *YOLO: State-of-the-Art Real-Time Object Detection*. [Ultralytics Docs](https://docs.ultralytics.com/).
- Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*. Cambridge University Press.
