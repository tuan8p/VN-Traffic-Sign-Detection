# VN-Traffic-Sign-Detection

Course project (IPCV): Vietnamese traffic-sign **object detection** with two pipelines, **Faster R-CNN (ResNet-50)** and **YOLO26m**.

Learning scaffold in the same layout as VN-Traffic-Sign-Classification. Shared offline preprocessing and data analysis; online preprocessing is an optional on/off switch while tuning each pipeline.

## Ai làm phần nào

Bốn người. Mỗi TODO ghi tên team. Danh sách file: TEAM_CHECKLIST.md.

- **Team analysis + preprocessing offline**: analysis/, vn_tsd/data/preprocess_offline.py, notebook 01 và 02.
- **Team pipeline YOLO26m + preprocessing online**: vn_tsd/pipelines/yolo/, configs/pipelines/yolo26m.yaml, notebook 04, nhánh eval YOLO.
- **Team pipeline Faster R-CNN + preprocessing online**: vn_tsd/pipelines/faster_rcnn/, configs/pipelines/faster_rcnn.yaml, notebook 03, nhánh eval Faster R-CNN.
- **Team demo + phân tích hình học và cấu trúc cảnh**: demo/ (kể cả demo/geometry.py), notebook 05. Hình học và cấu trúc cảnh theo mục 3.2 đề tổng hợp. Demo vẽ box từ checkpoint, không train model.
## Phân tích dữ liệu

Team analysis + preprocessing offline làm hai lần, cùng một bộ hình.

- Trước preprocessing offline: trên ảnh gốc và nhãn YOLO. Phân bố lớp, kích thước box, heatmap tâm box, số ảnh mỗi split, box nhỏ hơn min_box_px.
- Sau preprocessing offline: cùng các hình đó trên cache đã letterbox và augmentation, để xem bản dùng chung còn lệch lớp hoặc box không.

## Preprocessing offline

Ảnh sau bước này là chuẩn finetune chung cho cả Faster R-CNN và YOLO26m. Không crop thành từng biển.

- Letterbox 640, pad 114, ghi lại box YOLO theo ảnh mới.
- Có augmentation phù hợp trong cùng bước này. Không lật ngang, vì có biển đối xứng gương.
- Lưu images và labels cho train, val, test, kèm data.yaml và meta.json.
- Online preprocess không nằm ở đây. Từng team pipeline tự bật khi tune, và không sửa cache này.

## Sườn chung khi train

Cách dựng model và preprocessing online là việc của từng team. Phần dưới cả hai pipeline phải giữ.

- W&B bắt buộc.
- Hết train thì tự eval trên test. Eval test cũng chạy riêng bằng tools/run_eval.
- Seed lúc tìm config là 42. Khi chốt config, chạy thêm 7, 21, 123, 2026 rồi báo trung bình và độ lệch chuẩn.
- Full train một mạch từ weight pretrained, có warmup learning rate. Không làm hai stage khóa backbone rồi mở ra. Cả YOLO26m và Faster R-CNN dùng cùng kiểu này.
- Train trên Kaggle 2x T4, cùng kiểu DDP với pipeline DL bên classification: torchrun --nproc_per_node=2, AMP.
- Batch train và batch eval là tổng của cả hai GPU, và phải giống nhau ở cả hai pipeline. Chọn số sát VRAM của 2 T4 trên Kaggle để tận dụng. Team Faster R-CNN không đưa cả tổng đó vào mỗi GPU. Team YOLO đưa nguyên tổng cho Ultralytics.
- Team Faster R-CNN: data 52 lớp, head torchvision là 53 vì có background.
- Early stopping patience là 10, dùng chung như batch size. Đổi thì báo team kia.

- Metric chính sau mỗi epoch val, dùng để chọn best checkpoint: mAP50-95. Báo thêm mAP50, mAP75 và AR@100.

## Log lúc train

Cả hai pipeline gọi `vn_tsd/runtime/train_log.py`. Không tự in một format khác.

Trong epoch có thanh tqdm giống pipeline DL bên classification: `[train] Epoch  1/20`, postfix `loss`. Rank không phải main thì tắt thanh.

Hết epoch in một dòng, cùng thứ tự metric:

`[train] Epoch  1/20 | train_loss: 1.2346 | val_loss: 0.9877 | mAP50-95: 0.4100 | mAP50: 0.7200 | mAP75: 0.4500 | AR100: 0.5500 | lr: 5.00e-03`

Khi val mAP50-95 tốt hơn thì thêm `-> [BEST VAL mAP50-95: 0.4100]`. Early stop in `[train] early stopping at epoch 10 (patience=10)`.


## Lệnh torchrun

Kaggle 2× T4, chạy từ thư mục gốc repo. Batch trong config là tổng của hai GPU.

Faster R-CNN:

```bash
torchrun --nproc_per_node=2 -m tools.run_train --pipeline faster_rcnn --runtime configs/runtime/kaggle.yaml --run-name faster_rcnn_s42_ep20_bs8
```

YOLO26m:

```bash
torchrun --nproc_per_node=2 -m tools.run_train --pipeline yolo26m --runtime configs/runtime/kaggle.yaml --run-name yolo26m_s42_ep50_bs8
```

Mặc định là Kaggle 2× T4 với torchrun --nproc_per_node=2. Local chỉ để thử pipeline có chạy được không, dùng python -m tools.run_train và configs/runtime/local.yaml, không cố định số GPU. Thêm --data-root nếu cache không nằm ở data/processed. Thêm --online khi đang tune preprocessing online, và thêm _online vào tên run.

## W&B

Team [HCMUT_IPCV](https://forge.coreweave.com/wandb/HCMUT_IPCV), project `BTL`, host `https://forge.coreweave.com`.

Tên run hiện trên W&B, cả hai pipeline dùng một kiểu:

`{pipeline}_s{seed}_ep{epochs}_bs{batch_size}`

Ví dụ `yolo26m_s42_ep50_bs8` và `faster_rcnn_s42_ep20_bs8`. Bật online preprocess thì thêm `_online` ở cuối. Seed lúc tìm config là 42. Bốn seed còn lại chỉ đổi số sau `s`, giữ nguyên phần còn lại.


## Nhánh

Mỗi team tự tạo nhánh từ main và làm trên nhánh đó:

- preprocessing
- yolo
- faster-rcnn
- demo
## Layout

| Path | Role |
|------|------|
| `vn_tsd/data/preprocess_offline.py` | Shared cache (images + YOLO labels) |
| `vn_tsd/data/preprocess_online.py` | Optional per-batch toggles |
| `vn_tsd/pipelines/faster_rcnn/` | torchvision Faster R-CNN ResNet-50 |
| `vn_tsd/pipelines/yolo/` | Ultralytics YOLO26m |
| `analysis/` | Offline EDA |
| `demo/` | Streamlit demo |
| `configs/shared.yaml` | Locked shared keys |
| `configs/pipelines/` | Per-pipeline, including online toggles |

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -U pip && pip install -e ".[dev]"
# training extras when you fill the TODOs:
# pip install -e ".[detect]"
copy .env.example .env
```

## CLI

```bash
python -m analysis.eda --data-root <VNTS>
python -m tools.run_preprocess --data-root <VNTS>
python -m tools.run_train --pipeline faster_rcnn --runtime configs/runtime/local.yaml --run-name faster_rcnn_s42_ep20_bs8
python -m tools.run_train --pipeline yolo26m --runtime configs/runtime/local.yaml --run-name yolo26m_s42_ep50_bs8
torchrun --nproc_per_node=2 -m tools.run_train --pipeline faster_rcnn --runtime configs/runtime/kaggle.yaml --run-name faster_rcnn_s42_ep20_bs8
torchrun --nproc_per_node=2 -m tools.run_train --pipeline yolo26m --runtime configs/runtime/kaggle.yaml --run-name yolo26m_s42_ep50_bs8
python -m tools.run_eval --pipeline yolo26m --run-dir outputs/yolo26m/<run_id>
streamlit run demo/app.py
```

`--online` / `--no-online` override `online_preprocess.enabled` for that run. Individual ops (letterbox, mosaic, hflip, ...) only apply when the master switch is on. `hflip` stays false because several sign classes are mirror pairs.

## Outputs

Giống repo classification. Cuối train tự ghi run.zip trong thư mục run.

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
