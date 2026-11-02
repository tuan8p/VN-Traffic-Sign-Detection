# Demo — VN Traffic Sign Detection

**Owner:** Team demo + phân tích hình học và cấu trúc cảnh

## Cài đặt

```bash
# từ thư mục gốc repo
pip install -e ".[demo]"
# hoặc trực tiếp:
pip install streamlit>=1.28 pandas opencv-python-headless Pillow
```

## Chạy demo

```bash
streamlit run demo/app.py
```

Mở trình duyệt tại `http://localhost:8501`.

## Tính năng

| Tính năng | Mô tả |
|-----------|-------|
| **Upload ảnh** | Kéo thả hoặc chọn file (JPG, PNG, BMP, WebP) |
| **Ảnh mẫu có sẵn** | Tự động quét `demo/samples/`, hiển thị danh sách dạng list có số thứ tự, tìm kiếm nhanh và nút chuyển ảnh trước/sau |
| **Tự động nhận diện Checkpoint** | Tự động quét và nạp `checkpoint_best.pt` trong `outputs/faster_rcnn/` và `outputs/yolo26m/` (chưa có sẽ tự động dùng Mock) |
| **So sánh side-by-side** | 2 cột: Faster R-CNN vs YOLO26m, hiển thị ảnh vẽ box + metrics (thời gian, số box, avg conf, top-3 classes) |
| **Device selector** | Chọn trực quan giữa CPU hoặc CUDA (tự động nhận diện GPU khả dụng) |
| **Geometry 3.2** | Canny edge + HoughLinesP, overlay lên ảnh, tunable qua sidebar |

## Checkpoint

Demo **tự động quét** các thư mục đầu ra tiêu chuẩn của bài toán:

- Faster R-CNN: `outputs/faster_rcnn/**/checkpoint_best.pt`
- YOLO26m: `outputs/yolo26m/**/checkpoint_best.pt`

**Không cần nhập tay đường dẫn!**
- Nếu tìm thấy 1 file: tự động nạp.
- Nếu tìm thấy nhiều lần chạy (runs): hiển thị menu dropdown để chọn checkpoint mong muốn.
- Nếu chưa có: tự động kích hoạt chế độ **Mock predictions** để kiểm thử giao diện.

## Phân tích hình học — Mục 3.2

File: `demo/geometry.py`

**Hướng đã chọn:** Phát hiện biên + đường thẳng (Canny + HoughLinesP).

Pipeline:
1. Grayscale → Gaussian Blur (5×5)
2. Canny edge detection (ngưỡng tunable)
3. Probabilistic Hough Transform → line segments
4. Ba overlay: edges only, lines only, combined

Tất cả tham số điều chỉnh được qua sidebar.

## Files

| File | Vai trò |
|------|---------|
| `demo/app.py` | Streamlit UI chính |
| `demo/inference.py` | Load checkpoint, chạy inference, vẽ box |
| `demo/geometry.py` | Canny + HoughLinesP (mục 3.2) |
| `demo/samples/` | Ảnh mẫu để test nhanh |
