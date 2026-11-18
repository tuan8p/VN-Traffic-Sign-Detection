# Interactive Web Demo & Geometric Scene Analysis

Mô-đun demo trực quan hóa tương tác của dự án **VN-Traffic-Sign-Detection**, được xây dựng trên nền tảng **Streamlit**. Ứng dụng tích hợp đồng thời việc so sánh thực nghiệm hai mô hình Deep Learning (**Faster R-CNN** vs **YOLO26m**) và triển khai các giải thuật **Phân tích hình học & cấu trúc cảnh (Mục 3.2)**.

---

## 1. Các Tính Năng Nổi Bật

### A. So Sánh Hai Mô Hình Đối Sánh (Dual-Model Inference)
- **So sánh song song (Side-by-side)**: Trực quan hóa kết quả phát hiện vật thể của **Faster R-CNN (ResNet-50)** và **YOLO26m** trên cùng một khung hình.
- **Thẻ chỉ số hiệu năng thời gian thực**:
  - Thời gian suy luận (*Inference latency* tính bằng ms).
  - Số lượng bounding box phát hiện được.
  - Top-3 nhãn biển báo có độ tự tin cao nhất.
- **Tự động nhận diện trọng số (Auto Checkpoint Discovery)**: Tự động quét và nạp `checkpoint_best.pt` trong thư mục `outputs/`. Khi chưa có checkpoint thật, hệ thống tự động kích hoạt chế độ giả lập (*Mock mode*) phục vụ kiểm thử giao diện.
- **Bộ nhớ đệm mô hình (In-memory Model Caching)**: Mô hình chỉ nạp vào RAM một lần duy nhất, giúp các lần suy luận sau phản hồi tức thì (~0.1s – 0.4s).
- **Lựa chọn thiết bị tính toán**: Cho phép chuyển đổi linh hoạt giữa CPU và CUDA GPU (tự động fallback về CPU nếu GPU không khả dụng).

### B. Nạp Dữ Liệu Đa Dạng & An Toàn
- **Tải tệp ảnh linh hoạt**: Hỗ trợ kéo thả hoặc tải lên các định dạng phổ biến (`JPG`, `JPEG`, `PNG`, `BMP`, `WebP`, `JFIF`).
- **Xử lý ảnh an toàn**: Tự động chuẩn hóa góc xoay theo thẻ EXIF (*Auto-orient*), tự động phối màu nền trắng đối với ảnh PNG có kênh trong suốt (Alpha channel).
- **Thư viện ảnh mẫu có sẵn**: Tự động quét thư mục `demo/samples/`, hiển thị danh sách có thứ tự, kích thước tệp và nút điều hướng chuyển ảnh nhanh (*Ảnh trước / Ảnh sau*).

### C. Phân Tích Hình Học & Cấu Trúc Cảnh (Mục 3.2 Đề Bài)
Ứng dụng chia làm 2 tab chuyên sâu:

1. **Mục 3.2.A: Cấu trúc biên & đường thẳng toàn cảnh (Canny + Hough)**
   - Tiền xử lý lọc nhiễu Gaussian Blur ($\text{kernel} = 5$).
   - Dò biên Canny với thanh trượt điều chỉnh ngưỡng *Low* và *High*.
   - Trích xuất các đoạn thẳng cấu trúc bằng biến đổi Hough xác suất (*Probabilistic Hough Transform* — `cv2.HoughLinesP`).
   - 3 chế độ hiển thị overlay: *Edges* (biên trên nền xám), *Lines* (đoạn thẳng trên ảnh màu gốc), *Combined* (kết hợp cả hai).
   - Bảng thống kê định lượng: Số pixel biên, mật độ biên (%), số đoạn thẳng và độ dài trung bình.

2. **Mục 3.2.B: Nắn thẳng phối cảnh 2D Homography (Perspective Rectification)**
   - Tự động trích xuất vùng quan tâm (ROI) từ các biển báo phát hiện được (có thanh trượt mở rộng padding an toàn).
   - Giải thuật định vị 4 đỉnh góc của biển báo trong ROI (kết hợp xấp xỉ đa giác `approxPolyDP` và khung tứ giác nghiêng tối thiểu `minAreaRect`).
   - Ước lượng ma trận biến đổi phối cảnh 2D Homography $\mathbf{H} \in \mathbb{R}^{3 \times 3}$ từ 4 cặp điểm tương ứng bằng thuật toán Direct Linear Transformation (DLT).
   - Nắn thẳng biến dạng góc nghiêng (`cv2.warpPerspective`) đưa biển báo về góc nhìn trực diện chính diện (*Frontal View*) kích thước chuẩn $240 \times 240$ px.
   - Hiển thị ma trận $\mathbf{H}$ dưới dạng bảng số thực và công thức toán học KaTeX.

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

---

## 2. Cấu Trúc Thư Mục `demo/`

```
demo/
├── app.py          # Giao diện người dùng chính (Streamlit UI, quản lý state và layout)
├── geometry.py     # Triển khai thuật toán Canny, HoughLinesP và 2D Homography Rectification
├── inference.py    # Nạp checkpoint, pipeline inference (Faster R-CNN, YOLO26m), vẽ bounding box
├── samples/        # Thư mục chứa các tệp ảnh mẫu thực tế để kiểm thử nhanh
└── README.md       # Tài liệu kỹ thuật mô-đun demo
```

---

## 3. Hướng Dẫn Cài Đặt & Khởi Chạy

### Cài đặt môi trường
Từ thư mục gốc của repository:

```bash
# Cài đặt gói mở rộng cho demo
pip install -e ".[demo]"

# Hoặc cài đặt trực tiếp các thư viện phụ thuộc:
pip install streamlit>=1.28 pandas opencv-python-headless Pillow
```

### Khởi động ứng dụng

```powershell
# Chạy demo qua Streamlit
streamlit run demo/app.py

# Hoặc chỉ định tường minh Python 3.12:
& "C:\Users\LENOVO\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run demo/app.py
```

Sau khi khởi chạy thành công, giao diện web sẽ tự động mở tại địa chỉ `http://localhost:8501`.
