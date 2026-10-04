# Contributing

Phân công file: TEAM_CHECKLIST.md. Mỗi TODO ghi tên team. Đừng sửa phần team khác.

- Do not change `configs/shared.yaml` image size, class table, or split policy without agreeing it applies to both pipelines.
- Pipeline-only knobs live in `configs/pipelines/*.yaml`, including `online_preprocess`.
- Do not commit `data/raw`, `data/processed`, checkpoints, or `.env`.

## Nhánh

Mỗi team tự tạo nhánh từ main, làm trên nhánh đó, không commit thẳng vào main:

- preprocessing: analysis và preprocessing offline
- yolo: pipeline YOLO26m và preprocessing online của pipeline đó
- faster-rcnn: pipeline Faster R-CNN và preprocessing online của pipeline đó
- demo: demo và phân tích hình học, cấu trúc cảnh

Sườn train chung nằm ở configs/shared.yaml: W&B, eval test, seed, batch 2 GPU, early_stopping_patience, metric, zip. Đừng đổi batch_size, eval batch_size, hay early_stopping_patience trong file pipeline. Zip nằm ở vn_tsd/runtime/pack.py.