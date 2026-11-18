# Faster R-CNN (ResNet-50 FPN) Pipeline

Tài liệu kỹ thuật mô tả kiến trúc mô hình, hàm mất mát, chiến lược tiền xử lý và quy chuẩn huấn luyện của pipeline **Faster R-CNN** trong dự án nhận diện biển báo giao thông Việt Nam (**VN-Traffic-Sign-Detection**).

---

## 1. Kiến Trúc Mô Hình (Architecture)

Mô hình được xây dựng dựa trên kiến trúc hai giai đoạn (*Two-Stage Object Detector*), tận dụng trọng số tiền huấn luyện (*Pretrained Weights*) từ tập dữ liệu COCO và thực hiện Full Fine-tuning trên tập dữ liệu biển báo Việt Nam.

```text
Input (RGB 640×640)
       ↓
[ResNet-50 Backbone] + [Feature Pyramid Network (FPN)]
       ↓
[Region Proposal Network (RPN)] ──→ Đề xuất vùng ứng viên (RoI Proposals)
       ↓
[RoIAlign Layer] ──→ Trích xuất đặc trưng vùng (7×7 feature maps)
       ↓
[TwoMLPHead] ──→ Feature vector 1024 chiều
       ↓
[Custom MLP Predictor Head]
       ├── Linear(1024 → 512) → ReLU → Dropout(0.4)
       ├── Linear(512 → 53)  (Classification Logits: 52 lớp biển báo + 1 background)
       └── Linear(512 → 212) (Bounding Box Regression Offsets: 4 giá trị × 53 lớp)
```

### Chi tiết các khối thành phần:
1. **Backbone ResNet-50 + FPN**: Trích xuất biểu diễn đặc trưng ở nhiều cấp độ phân giải khác nhau, giúp mô hình nắm bắt tốt cả các biển báo kích thước nhỏ ở xa lẫn biển báo lớn ở cự ly gần.
2. **RPN & RoIAlign**:
   - RPN đề xuất các vùng chứa vật thể dựa trên hệ thống Anchor Box đa kích thước và tỷ lệ khung hình.
   - Lớp **RoIAlign** sử dụng nội suy song tuyến tính (*Bilinear Interpolation*) để lấy mẫu đặc trưng chính xác tại tọa độ thực, loại bỏ sai số lượng tử hóa của phép pooling thông thường.
3. **Cấu hình Head Predictor**:
   - Hỗ trợ 2 chế độ cấu hình qua YAML (`model.head`):
     - `standard`: Predictor chuẩn của TorchVision với 1 lớp Linear trực tiếp từ 1024 chiều.
     - `custom_mlp` (Cấu hình tối ưu mặc định): Bổ sung lớp chiếu giảm chiều trung gian Linear (1024 $\to$ 512), kích hoạt ReLU và lớp Dropout ($p = 0.4$) giúp chống hiện tượng quá khớp (*Overfitting*) trên các lớp có ít mẫu.

---

## 2. Hàm Mất Mát Đa Nhiệm (Multi-task Loss Formulation)

Quá trình tối ưu hóa sử dụng tổng có trọng số của 4 thành phần hàm mất mát:

$$\mathcal{L}_{total} = \mathcal{L}_{roi\_cls} + \mathcal{L}_{roi\_box} + \mathcal{L}_{rpn\_cls} + \mathcal{L}_{rpn\_box}$$

Trong đó:
- $\mathcal{L}_{roi\_cls}$: Hàm mất mát phân loại vật thể (Cross-Entropy Loss) trên các RoIs dương tính và âm tính.
- $\mathcal{L}_{roi\_box}$: Hàm mất mát hồi quy tọa độ hộp (Smooth L1 Loss) tính trên các RoIs dương tính.
- $\mathcal{L}_{rpn\_cls}$: Hàm mất mát xác định sự tồn tại của vật thể (Binary Cross-Entropy Loss) đối với từng Anchor.
- $\mathcal{L}_{rpn\_box}$: Hàm mất mát hồi quy tọa độ ban đầu của Anchor về vùng ứng viên (Smooth L1 Loss).

---

## 3. Tiền Xử Lý & Tăng Cường Dữ Liệu (Data Augmentation)

### A. Chuẩn hóa Offline (Shared Cache)
- Ảnh đầu vào được điều chỉnh kích thước về chuẩn $640 \times 640$ px bằng phép biến đổi Letterbox (giữ nguyên tỷ lệ khung hình và chèn đệm màu xám $\text{pad} = 114$).
- Chuẩn hóa cường độ điểm ảnh về $[0, 1]$ và áp dụng chuẩn hóa ImageNet ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Không sử dụng phép lật ngang (Horizontal Flip)**: Do nhiều cặp biển báo trong hệ thống biển báo Việt Nam mang tính đối xứng gương nhưng khác nghĩa (ví dụ: Biển cấm rẽ trái và Biển cấm rẽ phải).

### B. Tăng cường Online (Online Augmentation)
Trong quá trình huấn luyện, pipeline hỗ trợ các phép biến đổi ngẫu nhiên theo từng batch (`configs/pipelines/faster_rcnn.yaml`):
- **Affine Jitter**: Dịch chuyển, co giãn nhẹ với tỷ lệ jitter 0.1.
- **Color Jitter**: Tinh chỉnh độ sáng, độ tương phản và độ bão hòa màu để mô phỏng sự thay đổi thời tiết, ánh sáng ban ngày và ban đêm.
- **Mosaic & MixUp**: Kết hợp nhiều ảnh nhằm tăng độ phong phú ngữ cảnh cho vật thể.

---

## 4. Tham Số Huấn Luyện Tối Ưu (Optimal Hyperparameters)

Cấu hình chiến thắng được đúc kết từ quá trình tìm kiếm siêu tham số và kiểm chứng thực nghiệm:

| Siêu tham số | Giá trị cấu hình | Mô tả |
|---|---|---|
| **Optimizer** | SGD (Stochastic Gradient Descent) | Momentum = 0.9 |
| **Learning Rate (LR)** | $9.606 \times 10^{-3}$ | Learning rate cơ sở sau quá trình tuning |
| **Weight Decay** | $1.476 \times 10^{-5}$ | Hệ số suy giảm trọng số L2 |
| **LR Scheduler** | Cosine Annealing | Có 3 epoch khởi động mềm (Warmup) từ hệ số $0.001$ |
| **Global Batch Size** | 24 | Tổng batch trên 2 GPU (12 mẫu / GPU trên Kaggle 2× T4) |
| **Head Architecture** | `custom_mlp` | Hidden Dim = 512, Dropout = 0.4 |
| **Patience** | 10 epochs | Early stopping nếu `mAP50-95` trên tập validation không tăng |
| **Max Epochs** | 100 | Giới hạn số chu kỳ huấn luyện tối đa |

---

## 5. Quy Chuẩn Đánh Giá (Evaluation Protocol)

Đánh giá được thực hiện độc lập trên tập `test` bằng bộ đánh giá tiêu chuẩn **COCOeval API** (`pycocotools.COCOeval`):
- **mAP50-95** (Chỉ số đánh giá chính): Độ chính xác trung bình trung bình tính trên các ngưỡng IoU từ 0.50 đến 0.95 với bước nhảy 0.05.
- **mAP50 / mAP75**: Độ chính xác tại các ngưỡng IoU cụ thể 0.50 và 0.75.
- **AR@100**: Độ thu hồi trung bình (Average Recall) với tối đa 100 detections trên mỗi ảnh.
- **Kiểm định đa seed**: Để loại trừ yếu tố may rủi do khởi tạo ngẫu nhiên, mô hình được đánh giá trên 5 seed độc lập: `[42, 7, 21, 123, 2026]`, kết quả được báo cáo dưới dạng giá trị trung bình kèm độ lệch chuẩn ($\mu \pm \sigma$).

---

## 6. Hướng Dẫn Thực Thi CLI

```bash
# 1. Huấn luyện cục bộ trên CPU / 1 GPU:
python -m tools.run_train \
    --pipeline faster_rcnn \
    --runtime configs/runtime/local.yaml \
    --run-name frcnn_local_exp

# 2. Huấn luyện phân tán đa GPU (2× T4 trên Kaggle):
torchrun --nproc_per_node=2 -m tools.run_train \
    --pipeline faster_rcnn \
    --runtime configs/runtime/kaggle.yaml \
    --run-name frcnn_kaggle_s42

# 3. Đánh giá một checkpoint đã lưu trên tập test:
python -m tools.run_eval \
    --pipeline faster_rcnn \
    --run-dir outputs/faster_rcnn/<run_id>
```
