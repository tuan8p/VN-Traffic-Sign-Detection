# Thống nhất metric YOLO và Faster R-CNN

## Contract đã hiện thực

- Metric chính: `mAP50-95`; các metric bổ sung: `mAP50`, `mAP75`, `AR100`.
- `evaluate()` trả dict phẳng gồm đúng bốn khóa, giá trị trong [0, 1].
- Chưa tính được dùng `None` (JSON `null`), log `N/A`. Số 0 chỉ dùng khi thực sự đo được 0.
- Dùng `vn_tsd.eval.metrics.summarize(raw)` để chuẩn hóa và kiểm tra giá trị.
- Validation chọn checkpoint, test chỉ đánh giá checkpoint đã chọn. Không tune bằng test.
- Không thay đổi cấu hình .env hoặc YAML trong bước thống nhất schema này.

## Log train (history.json + stdout)

Mỗi epoch, rank 0 ghi một dòng qua `vn_tsd.runtime.train_log.log_epoch` và append
`history_row(...)` vào `history.json`. Khóa bắt buộc: `epoch`, `train_loss`,
`val_loss`, `lr`, `is_best`, `patience_count`, `mAP50-95`, `mAP50`, `mAP75`, `AR100`.
W&B dùng `wandb_epoch_payload()` (`train_loss`, `val_loss`, `lr`, `val/mAP50-95`, …).

YOLO: `train_loss` = tổng loss train native; `val_loss` = `val/box_loss` Ultralytics.

## Trạng thái YOLO (sau thống nhất COCOeval)

- **Chọn best + early stop:** val **COCOeval** qua `evaluate_split_coco(..., split="val")` trên `trainer.last` mỗi epoch; metric chính `mAP50-95`.
- **Test / `run_eval`:** cùng `pycocotools.COCOeval bbox` như Faster R-CNN (`vn_tsd/eval/coco_eval.py`), postprocess `score_thresh`/`nms_thresh`/`detections_per_img` khớp yaml.
- **`test_metrics.json`:** `metrics` = COCO; `metrics_native` = Ultralytics (tham khảo, không dùng chọn model).
- **`metrics.json`:** `metric_source: pycocotools.COCOeval bbox`, cùng khung với Faster R-CNN.

## Đề xuất trao đổi với team Faster R-CNN

1. Trả cùng schema từ `evaluate()`; gọi `summarize()` và giữ metric thiếu là null.
2. Khi dùng COCOeval bbox, lấy AP/AP50/AP75 và AR tại maxDets=100, area=all.
   AR100 trung bình trên IoU 0.50:0.05:0.95, không phải recall tại một ngưỡng confidence.
   Giá trị COCO không xác định (ví dụ -1) cần chuyển thành None trước summarize().
3. Thống nhất mapping class: YOLO 0..51; R-CNN bỏ background và chuyển label
   về cùng category ID. Ground truth và prediction dùng cùng tọa độ pixel, bbox xywh.
4. Cùng cache, split, tập lớp; giữ ảnh không có detection trong tập đánh giá.
5. Để so sánh cuối cùng, đề xuất dùng chung COCOeval cho cả hai pipeline,
   maxDets=[1, 10, 100], cùng policy lọc score và hậu xử lý được ghi rõ.
   Chưa coi native Ultralytics AP và COCO AP là hoàn toàn tương đương.
6. Khi chuyển sang evaluator chung, chốt riêng cách tích hợp validation để cả
   hai pipeline chọn checkpoint bằng cùng định nghĩa mAP50-95. Hiện chưa xác minh
   fitness native của phiên bản Ultralytics đang dùng chỉ phụ thuộc metric này.

Team R-CNN chưa cần sửa training loop để nhận schema: CLI dùng chung đã gọi
`pipeline.evaluate()`. Phần R-CNN hiện vẫn scaffold; tài liệu này là đề xuất bàn giao,
không xác nhận evaluation R-CNN đã chạy được.
