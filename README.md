# VN-Traffic-Sign-Detection

Vietnamese traffic-sign **object detection** with two pipelines: **Faster R-CNN (ResNet-50)** and **YOLO26m**.

Project covers:
- **Data analysis & exploration (EDA)**: class distribution, bounding box size/aspect ratios, heatmap analysis, cluster visualization.
- **Offline & online preprocessing**: 640×640 letterbox, padding, mosaic, mixup, and class-preserving augmentations.
- **Model training & evaluation**: multi-seed evaluation (42, 7, 21, 123, 2026), DDP on multi-GPU, COCO evaluation metrics (mAP50-95, mAP50, mAP75, AR100).
- **Geometric scene analysis (Section 3.2)**: Canny edge detection + Probabilistic Hough Lines, 2D Homography Perspective Rectification.
- **Interactive web demo (Section 3.4)**: Streamlit application comparing Faster R-CNN vs. YOLO26m with real checkpoints, side-by-side inference, and geometry analysis tools.

## Cấu trúc dự án (Layout)

| Thư mục / File | Vai trò |
| ------ | ------ |
| `vn_tsd/data/preprocess_offline.py` | Tiền xử lý dữ liệu offline (letterbox 640×640, YOLO labels, augmentation) |
| `vn_tsd/data/preprocess_online.py` | Cấu hình bật/tắt online augmentation khi huấn luyện |
| `vn_tsd/pipelines/faster_rcnn/` | Pipeline Faster R-CNN ResNet-50 FPN (TorchVision) |
| `vn_tsd/pipelines/yolo/` | Pipeline YOLO26m (Ultralytics) |
| `analysis/` | Mô-đun phân tích dữ liệu EDA (biểu đồ lưu tại `analysis/figures/`) |
| `demo/` | Ứng dụng Streamlit demo và phân tích hình học (Mục 3.2 & 3.4) |
| `configs/shared.yaml` | Cấu hình tham số chung cho cả 2 pipeline (batch size, seeds, metrics) |
| `configs/pipelines/` | Cấu hình riêng cho từng mô hình (`faster_rcnn.yaml`, `yolo26m.yaml`) |
| `configs/runtime/` | Cấu hình môi trường thực thi (local, Kaggle) |
| `tools/` | CLI công cụ: `run_preprocess.py`, `run_train.py`, `run_eval.py`, `pack_outputs.py` |

## Tiền xử lý và huấn luyện

### 1. Phân tích dữ liệu (EDA)
Thực hiện phân tích phân bố lớp, kích thước bounding box, heatmap vị trí tâm và độ sắc nét/độ sáng của tập dữ liệu trước và sau tiền xử lý để bảo đảm tính cân bằng.

### 2. Tiền xử lý dữ liệu offline
Ảnh sau bước này là chuẩn huấn luyện chung cho cả Faster R-CNN và YOLO26m:
- Letterbox 640×640, pad 114, chuẩn hóa nhãn bounding box YOLO.
- Augmentation phù hợp, không lật ngang (hflip) để tránh thay đổi ý nghĩa các biển báo có tính đối xứng gương.
- Xuất dữ liệu `train`, `val`, `test` kèm `data.yaml` và `meta.json`.

### 3. Quy chuẩn huấn luyện
- Đăng nhập và theo dõi tiến trình qua Weights & Biases (W&B).
- Đánh giá trên tập `test` sau khi huấn luyện xong bằng `tools/run_eval.py`.
- Seed chuẩn: 42 khi tinh chỉnh siêu tham số; báo cáo trung bình và độ lệch chuẩn trên 5 seed: `[42, 7, 21, 123, 2026]`.
- Huấn luyện trên GPU với DDP và AMP (`torchrun --nproc_per_node=2`).
- Tiêu chí đánh giá chính: **mAP50-95** (COCOeval), kèm mAP50, mAP75 và AR@100.

## Hướng dẫn cài đặt (Quick start)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -U pip && pip install -e ".[dev]"
# Thêm các thư viện nhận diện & huấn luyện:
pip install -e ".[detect]"
copy .env.example .env
```

## Dòng lệnh CLI

```bash
# 1. Chạy phân tích dữ liệu EDA
python -m analysis.eda --data-root <VNTS>

# 2. Tiền xử lý offline
python -m tools.run_preprocess --data-root <VNTS>

# 3. Huấn luyện cục bộ (Local)
python -m tools.run_train --pipeline faster_rcnn --runtime configs/runtime/local.yaml --run-name faster_rcnn_s42_ep20_bs8
python -m tools.run_train --pipeline yolo26m --runtime configs/runtime/local.yaml --run-name yolo26m_s42_ep50_bs8

# 4. Huấn luyện phân tán (Kaggle 2× T4 GPU)
torchrun --nproc_per_node=2 -m tools.run_train --pipeline faster_rcnn --runtime configs/runtime/kaggle.yaml --run-name faster_rcnn_s42_ep20_bs8
torchrun --nproc_per_node=2 -m tools.run_train --pipeline yolo26m --runtime configs/runtime/kaggle.yaml --run-name yolo26m_s42_ep50_bs8

# 5. Đánh giá checkpoint độc lập
python -m tools.run_eval --pipeline faster_rcnn --run-dir outputs/faster_rcnn/<run_id>
python -m tools.run_eval --pipeline yolo26m --run-dir outputs/yolo26m/<run_id>

# 6. Khởi động Web Demo Streamlit
streamlit run demo/app.py
```

## Kết quả xuất ra (Outputs)

Mỗi lần huấn luyện hoàn tất sẽ tự động đóng gói file `run.zip` và lưu trữ tại:

```
outputs/<pipeline>/<run_id>/
  resolved_config.yaml
  metrics.json
  history.json
  figures/
  checkpoints/checkpoint_best.pt
  run.zip
```

## License

MIT
