# Phân công 4 người

Mỗi dòng TODO ghi tên team. Chỉ sửa phần của mình. File ghi dùng chung thì hai team pipeline đều đọc, mỗi team chỉ điền nhánh của mình.

## Team analysis + preprocessing offline

- analysis/eda.py, analysis/README.md, notebooks/01_data_analysis.ipynb
- vn_tsd/data/preprocess_offline.py, tools/run_preprocess.py, notebooks/02_preprocess_offline.ipynb
- docs/DATA_CONTRACT.md (layout data/processed)
- Giữ box YOLO, không crop thành ảnh phân loại. Viết images + labels cho train/val/test, data.yaml, meta.json.

## Team pipeline YOLO26m + preprocessing online

- vn_tsd/pipelines/yolo/, configs/pipelines/yolo26m.yaml, notebooks/04_train_yolo26m.ipynb
- Full train một mạch, có warmup learning rate. Không hai stage.
- Đưa đúng batch tổng trong shared.yaml cho Ultralytics. Không chia. Cùng số với Faster R-CNN, chọn sát VRAM của 2 T4.
- tools/run_eval.py và vn_tsd/eval/metrics.py: nhánh Ultralytics
- Online preprocess bật/tắt bằng online_preprocess.enabled (CLI --online / --no-online). Khi bật, map mosaic, mixup, jitter, color_jitter sang hyp Ultralytics trong train loop. Không sửa cache offline. hflip mặc định tắt.

## Team pipeline Faster R-CNN + preprocessing online

- vn_tsd/pipelines/faster_rcnn/, configs/pipelines/faster_rcnn.yaml, notebooks/03_train_faster_rcnn.ipynb
- Full train một mạch, có warmup learning rate. Không hai stage.
- DataLoader dùng per_rank_batch để tổng trên 2 GPU vẫn bằng batch trong shared.yaml, không nhân đôi. Head là 53 lớp (52 + background).
- tools/run_eval.py và vn_tsd/eval/metrics.py: nhánh torchvision
- Online preprocess dùng contract vn_tsd/data/preprocess_online.py (hàm resolve đã có). Áp flag trong dataloader / train loop. Không sửa cache offline.

## Team demo + phân tích hình học và cấu trúc cảnh

Theo mục 3.2 trong CV-project-tonghop.pdf, làm ít nhất một hướng:

- Phát hiện biên, đường thẳng hoặc góc (edges, lines, corners).
- Ước lượng biến đổi hình học affine hoặc projective.
- Ghép ảnh (panorama) hoặc căn chỉnh nhiều ảnh cùng một cảnh.
- Ước lượng mặt phẳng hoặc vùng bề mặt chính.

Và demo (mục 3.4, cộng demo tương tác):

- demo/app.py, demo/inference.py, demo/geometry.py, demo/README.md, notebooks/05_demo.ipynb
- Vẽ bounding box từ checkpoint của hai pipeline, không train lại model.
- Chồng kết quả hình học lên ảnh (và video nếu có), so sánh ảnh vào và ảnh ra.

## Dùng chung, không thuộc một team

- vn_tsd/config, vn_tsd/utils, vn_tsd/runtime (kể cả runtime/train_log.py), configs/shared.yaml, configs/classes.csv, tools/run_train.py
- vn_tsd/data/preprocess_online.py chỉ resolve flag. Logic augmentation nằm trong train loop của team YOLO26m và team Faster R-CNN.

## Nhánh mỗi team tự tạo

- preprocessing
- yolo
- faster-rcnn
- demo

Sườn train chung (W&B, tự eval test, seed, batch, early stopping patience, metric, zip) nằm trong README và configs/shared.yaml. Hai team pipeline không tự đặt batch train, eval batch, hay early_stopping_patience khác nhau.
Zip không thuộc team nào: vn_tsd/runtime/pack.py và tools/pack_outputs.py đóng run.zip cho mọi pipeline.
