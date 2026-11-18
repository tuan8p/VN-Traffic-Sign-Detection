# Exploratory Data Analysis (EDA) — Phân Tích Dữ Liệu

Thư mục này chứa toàn bộ mã nguồn và biểu đồ phục vụ việc phân tích, thăm dò và kiểm định chất lượng bộ dữ liệu biển báo giao thông Việt Nam trước và sau khi tiền xử lý.

---

## 1. Mục Tiêu & Phương Pháp Luận

Để đảm bảo mô hình huấn luyện hội tụ ổn định và không bị thiên lệch, quy trình EDA được thực hiện ở **hai giai đoạn**:
1. **Dữ liệu gốc (Raw Data)**: Đánh giá đặc điểm tự nhiên của ảnh và nhãn bounding box (phân bố lớp, độ phân giải, tỷ lệ khung hình, độ sắc nét, độ sáng).
2. **Dữ liệu sau xử lý (Processed Cache)**: Kiểm chứng lại toàn bộ các thuộc tính trên tập dữ liệu đã qua chuẩn hóa (Letterbox 640×640, Padding, Data Augmentation) nhằm đảm bảo không phát sinh rò rỉ dữ liệu (data leakage) hay méo mó nhãn.

---

## 2. Danh Mục Các Biểu Đồ Phân Tích

Toàn bộ biểu đồ phân tích được tự động kết xuất vào thư mục `analysis/figures/raw/` và `analysis/figures/processed/`:

| Tên biểu đồ | Nội dung & Ý nghĩa phân tích |
|---|---|
| `01_class_distribution.png` | **Phân bố số lượng bounding box theo 52 lớp biển báo**: Nhận diện mức độ mất cân bằng dữ liệu (class imbalance) giữa các lớp phổ biến và các lớp hiếm. |
| `02_boxes_per_image.png` | **Số lượng biển báo xuất hiện trên mỗi ảnh**: Thống kê mật độ vật thể trong môi trường giao thông thực tế. |
| `03_box_dimensions.png` | **Kích thước và tỷ lệ khung hình (Aspect Ratio) của box**: Phân tích phân bố chiều rộng/chiều cao để tối ưu kích thước anchor box cho RPN. |
| `04_box_center_heatmap.png` | **Bản đồ nhiệt (Spatial Heatmap) vị trí tâm biển báo**: Đánh giá sự phân bố không gian (thường tập trung ở nửa trên và hai bên lề đường). |
| `05_images_per_split.png` | **Số lượng ảnh phân chia theo từng tập**: Kiểm tra tỷ lệ phân chia tập dữ liệu `train` (70%), `val` (15%), `test` (15%). |
| `06_tiny_box_candidates.png` | **Phân tích các biển báo kích thước siêu nhỏ (< 8px)**: Sàng lọc các vật thể quá nhỏ hoặc nhãn nhiễu để có chiến lược xử lý phù hợp. |
| `07_data_leakage.png` | **Kiểm định trùng lặp và rò rỉ dữ liệu**: Ứng dụng giải thuật sai phân băm ảnh (Difference Hash - dHash) với khoảng cách Hamming để phát hiện ảnh chụp liên tiếp hoặc trùng lặp giữa tập train và test. |
| `08_brightness_sharpness.png` | **Độ sáng và độ sắc nét ảnh**: Đo lường giá trị trung bình mức xám và phương sai toán tử Laplacian (Laplacian Variance) nhằm đánh giá ảnh mờ, tối hoặc chói sáng. |

---

## 3. Công Cụ Trực Quan Hóa Tương Tác (GUI Tools)

Bên cạnh các script xuất biểu đồ tĩnh, mô-đun cung cấp 2 ứng dụng giao diện trực quan hỗ trợ rà soát nhanh dữ liệu:

1. **Trình duyệt lớp biển báo (`analysis/class_viewer_gui.py`)**:
   - Cho phép duyệt qua từng lớp trong 52 lớp biển báo.
   - Lọc ảnh theo số lượng mẫu (lớp ít mẫu, lớp trung bình, lớp nhiều mẫu).
   - Hiển thị trực tiếp ảnh và khung bounding box tương ứng.
   ```bash
   python -m analysis.class_viewer_gui
   ```

2. **Trình duyệt cụm ảnh tương đồng (`analysis/cluster_viewer_gui.py`)**:
   - Hiển thị các nhóm ảnh có độ tương đồng cao được gom cụm bằng thuật toán dHash.
   - Giúp phát hiện các chuỗi khung hình liên tiếp từ camera hành trình để tránh đưa vào cả train và test.
   ```bash
   python -m analysis.cluster_viewer_gui
   ```

---

## 4. Hướng Dẫn Sử Dụng Dòng Lệnh

Chạy quy trình phân tích tự động trên toàn bộ tập dữ liệu:

```bash
# Phân tích tập dữ liệu gốc
python -m analysis.eda --data-root <path_to_raw_dataset>

# Sau khi chạy xong, toàn bộ kết quả phân tích thống kê được tóm tắt tại:
# analysis/figures/raw/eda_summary.json
```
