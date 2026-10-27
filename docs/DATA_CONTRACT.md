# Data contract

Source is a YOLO detection set (images + labels), same family as VNTS (52 classes, `configs/classes.csv`). This repo does **not** crop boxes into classification samples.

## Expected raw layout

```
DATA_ROOT/
  images/          # file ảnh (.jpg, .png, ...)
  labels/          # YOLO txt: class cx cy w h (normalized 0-1)
  split_dataset/   # (tùy chọn) train.txt, val.txt, test.txt – mỗi dòng là tên ảnh
```

## Processed layout (offline)

Chạy `python -m tools.run_preprocess --data-root <DATA_ROOT>` để sinh ra cây này:

```
data/processed/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  data.yaml     # chuẩn Ultralytics: path, train, val, test, nc, names
  meta.json     # thống kê: số ảnh/box mỗi split, box bị loại, config dùng
```

### data.yaml format

```yaml
path: .
train: images/train
val: images/val
test: images/test
nc: 52
names:
  0: Đường người đi bộ cắt ngang
  1: Đường giao nhau (ngã ba bên phải)
  ...
  51: Chú ý chướng ngại vật – vòng tránh sang bên phải
```

### meta.json format

```json
{
  "config": {
    "target_size": [640, 640],
    "pad_value": 114,
    "min_box_px": 8,
    "num_classes": 52,
    "seed": 42
  },
  "splits": {
    "train": { "n_images": ..., "n_boxes_in": ..., "n_boxes_out": ..., "n_tiny_dropped": ... },
    "val":   { ... },
    "test":  { ... }
  },
  "total_dropped_tiny": ...,
  "total_boxes_kept": ...,
  "elapsed_seconds": ...
}
```

## Preprocessing steps

1. **Letterbox resize** – giữ tỉ lệ khung hình, resize vào 640×640, bù màu xám 114.
2. **Cập nhật tọa độ box** – tính lại `cx cy w h` normalized theo ảnh mới.
3. **Lọc tiny box** – bỏ box có `min(w,h) < 8px` sau letterbox.
4. **Split** – ưu tiên dùng `split_dataset/` có sẵn; nếu không, tự chia 70/15/15 ngẫu nhiên (seed 42).
5. **Ghi ảnh** – JPEG quality 95.
6. **Ghi nhãn** – YOLO `.txt` cùng stem với ảnh.

## Rules

- **Train / val / test** đều lấy từ cache preprocess dưới `data.root` (`data.yaml` + `images/*` + `labels/*`). Hai pipeline gọi `validate_processed_dataset()` trước train/eval; không dùng tập val/test bên ngoài.
- **Ngoại lệ duy nhất:** checkpoint pretrain (`FasterRCNN_ResNet50_FPN COCO_V1`, `yolo26m.pt`) tải từ nhà phát hành.
- Horizontal flip **mặc định tắt** (`hflip: false` trong shared.yaml) — biển báo giao thông Việt Nam có các cặp đối xứng gương mang ý nghĩa trái ngược.
- Primary metric: **mAP@0.50:0.95** (`mAP50-95`). Cũng báo `mAP50`, `mAP75`, `AR@100`.
- Cả hai pipeline (Faster R-CNN và YOLO26m) đọc **cùng một cây** `data/processed/`. Online aug là switch tune riêng, không sửa cache này.
- Không lưu crop ảnh phân loại (`.npy` hay thư mục per-class). Đây là bài toán **object detection**.
