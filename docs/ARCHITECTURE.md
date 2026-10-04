# Architecture

```
raw YOLO images+labels
        -> offline preprocess (shared, cached)
        -> data/processed + analysis figures
        ├─ Faster R-CNN ResNet-50   (optional online toggles)
        └─ YOLO26m                   (optional online toggles)
                 ↓
            outputs/<pipeline>/<run_id>/
                 ↓
            demo (streamlit)
```

Offline preprocessing is shared. Online preprocessing is per-pipeline and off unless `online_preprocess.enabled` is true.

Team trên sơ đồ:

- Offline preprocess và analysis: Team analysis + preprocessing offline.
- Nhánh YOLO26m, kể cả online preprocess của nhánh đó: Team pipeline YOLO26m + preprocessing online.
- Nhánh Faster R-CNN, kể cả online preprocess của nhánh đó: Team pipeline Faster R-CNN + preprocessing online.
- Demo và phân tích hình học / cấu trúc cảnh (mục 3.2 đề tổng hợp): Team demo + phân tích hình học và cấu trúc cảnh.
