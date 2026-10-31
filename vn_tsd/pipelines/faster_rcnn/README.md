# Faster R-CNN

Pipeline nhận **ảnh processed RGB 640×640 cùng YOLO labels**, trả boxes/class/scores trên tọa độ ảnh processed.
Không tạo cache offline, đổi split hoặc crop từng biển thành classification. Online augmentation chỉ áp vào train.
Implementation không đồng nghĩa đã có kết quả huấn luyện; random-weight smoke chỉ kiểm đường chạy.

## Phương pháp thực sự dùng

ResNet-50 trích đặc trưng; FPN ghép thông tin nhiều mức phân giải, giúp biểu diễn vật thể có nhiều kích thước.
RPN đề xuất vùng dựa trên anchors, dự đoán objectness và độ lệch box. RoIAlign lấy đặc trưng vùng với nội suy,
tránh lượng tử hóa vị trí như pooling đơn giản. RoI head dự đoán class và tinh chỉnh box; score filtering,
class-wise NMS và giới hạn số detection tạo output cuối. Giữ anchors/RPN/RoI sampling mặc định của TorchVision v1 baseline;
chưa có bằng chứng đây là lựa chọn tối ưu cho biển báo nhỏ.

Pretrained COCO V1 được nạp với 91 outputs rồi thay head thành **53**: background0 + dataset IDs0..51 được cộng1.
Mọi learnable parameter train ngay từ epoch1; FrozenBatchNorm giữ thống kê pretrained và eps0, không có lịch freeze/unfreeze hai stage.
Đây là full finetune parameter, không phải cập nhật lại running statistics của BatchNorm.

### Custom head và baseline chuẩn

Theo góp ý thêm layers của Tuấn, YAML hiện chọn `model.head: custom_mlp`. Giữ nguyên pretrained backbone/FPN/RPN và
hai FC layers của `roi_heads.box_head`; thay riêng predictor sau feature vector1024 bằng:

```text
RoIAlign → pretrained TwoMLPHead → feature1024
                                   ↓
                         Linear(1024 → 512)
                                   ↓
                                  ReLU
                                   ↓
                              Dropout(0.2)
                              ↙          ↘
                    Linear(512→53)   Linear(512→212)
                    class logits     4 box offsets/class
```

Các layer predictor mới được khởi tạo mới; COCO state được load strict trước khi thay predictor. Dropout chỉ hoạt động khi train;
inference và diagnostic val-loss tắt Dropout. Đây là một biến thể để thử nghiệm, chưa có kết quả chứng minh tăng mAP.
Đổi `model.head` sang `standard` giữ predictor TorchVision53 của baseline, không thêm projection/Dropout. Ví dụ override ablation:

```yaml
model:
  head: standard  # custom_mlp để dùng head thêm layers
```

`head_hidden_dim` (mặc định512) và `head_dropout` (0.2) chỉ có tác dụng với custom_mlp. Khi so sánh head chuẩn/custom,
giữ cùng cache, seed, batch, scheduler và online flags; chọn bằng validation. Không sửa notebook scaffold để thêm knobs riêng.
Checkpoint ghi loại head, version, hidden dimension và dropout để dựng lại chính xác; sai metadata/weights thì fail strict.
Checkpoint cũ không có trường head vẫn được hiểu là predictor chuẩn và giữ nguyên config hash/metadata của run cũ.

Loss tổng là `loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg`:

- RoI classification dùng cross entropy; box regression dùng Smooth L1 trên positive RoIs.
- RPN objectness dùng binary cross entropy; RPN regression dùng Smooth L1 trên positive anchors.

Chi tiết reduction/assignment/sampling theo source TorchVision được ghi phiên bản trong artifacts; không thay loss bằng công thức MixUp weighted classification.
Để tính diagnostic val loss, model cần detection train-mode trong `no_grad`; pipeline giữ BN/dropout eval và phục hồi RNG,
flags/buffers. COCO metric lấy từ pass inference riêng; **val loss không chọn best**.

Input luôn float32 CHW `[0,1]`; model normalize đúng một lần bằng ImageNet mean/std.
Cache đã letterbox640, vì vậy flag letterbox/normalize là yêu cầu đã được thỏa, không phải hai augmentation thêm.
Online ON chạy mosaic → affine jitter → color jitter → MixUp theo config; OFF bỏ optional ops. Hflip ON bị từ chối vì mirror classes.
Xem `transforms.py` để biết toán học/ordering; không sửa ảnh/label cache trên disk.

## Cache và nghiệm thu dữ liệu

```
processed/
  images/{train,val,test}/
  labels/{train,val,test}/
  data.yaml
  meta.json
```

`--data-root` có thẩm quyền hơn data.yaml.path trên máy build. Class names phải khớp `configs/classes.csv` theo cột name_vi và ID.
Ảnh phải decode được và 640×640, label file phải tồn tại (rỗng là negative hợp lệ). Mỗi dòng là integer class0..51 + normalized cx,cy,w,h.
Adapter kiểm finite/range/box suy biến, không sửa nhãn hoặc lọc test cho riêng FRCNN. Boxes nhỏ hơn min_box_px được báo và giữ;
cache owner phải xác minh policy offline trước run thật.

Dataset identity lưu sorted relative paths + SHA256 bytes ảnh/nhãn, metadata/class hashes và số lượng. Chặn ảnh byte-identical giữa split;
không chứng minh độc lập với ảnh re-encode/near-duplicate/augmentation cùng lineage. Cần cache final/version/provenance của Bình.
Các vấn đề ngoài ownership được ghi tại [file gửi nhóm](../../../docs/FRCNN_TEAM_HANDOFF.md).

## Local — Pixi và kiểm thử offline

Từ root repo, dùng môi trường `frcnn`, không dùng env thư mục cha:

```powershell
pixi install -e frcnn --locked
pixi run -e frcnn --locked python -m pytest -q
```

Heavy tests skip rõ nếu thiếu torch/torchvision/pycocotools; CI `.[dev]` chỉ kiểm phần nhẹ. Pixi suite kiểm thêm
model checkpoint thật, inference640, backward128, fixtures và orchestration hai CPU/Gloo ranks. Tiny model chỉ kiểm điều phối;
không chứng minh NCCL2×T4 hoặc chất lượng detection. Không test nào tải data/weights hoặc gọi W&B thật.

Nếu cần smoke bằng CLI trên cache hợp lệ, viết **file overrides riêng cho smoke**, không sửa shared config:

```yaml
model:
  pretrained: false
train:
  epochs: 1
  batch_size: 1
  require_wandb: false
eval:
  batch_size: 1
runtime_cfg:
  device: cpu
  num_workers: 0
  mixed_precision: false
outputs:
  zip_after_train: false
```

```powershell
pixi run -e frcnn --locked python -m tools.run_train --pipeline faster_rcnn --runtime configs/runtime/local.yaml --data-root D:/path/to/processed --overrides D:/path/to/smoke.yaml --run-name frcnn_smoke
```

CLI smoke này vẫn forward/train ở640 và có thể chậm/nặng trên laptop. Không chạy mặc định trong setup hoặc coi nó là full-run result.
Batch smoke khác batch nhóm được ghi trong resolved/effective config; không đưa số smoke vào bảng so sánh.

## Kaggle full training và eval

Notebook [03](../../../notebooks/03_train_faster_rcnn.ipynb) giữ nguyên toàn bộ scaffold của Tuấn trên main theo yêu cầu người dùng,
chỉ đổi REPO_BRANCH=faster-rcnn. Notebook dùng knobs/overrides, connectivity-test W&B và train launcher như scaffold;
không có cell preflight mới hoặc standalone eval bổ sung. Bootstrap fetch rồi checkout -B về remote khi người dùng chạy cell.
Branch remote phải chứa implementation trước khi dùng bootstrap; không chạy cell này trên local đang có code chưa push.
Scaffold cài detect/W&B nhưng chưa cài pycocotools: cần chuẩn bị dependency này riêng trước train/eval.
Knobs batch/patience/seed trong scaffold có thể override YAML; giữ chúng khớp shared config. Connectivity-test tạo W&B run riêng;
FRCNN fit vẫn chỉ init một training run ở rank0. Những giới hạn scaffold khác được ghi trong file gửi nhóm, chưa xác nhận notebook chạy Kaggle end-to-end.
Chưa hỗ trợ chạy lại vào run-dir đã có resolved_config/checkpoint; chọn run mới. V1 **không training resume**.

Global batch train/eval và patience đọc shared config; per-rank batch chia theo WORLD_SIZE thực.
Training DistributedSampler có thể padding sample khi N không chia hết world size (được log), tail batch không drop.
Val/test shard không pad/drop; unwrapped model forward, gather predictions một lần sau shard và rank0 COCOeval.
Rank0 chọn best strict val mAP50-95 tăng; score0 vẫn lưu checkpoint đầu, tie không overwrite, stop đúng patience.

Scheduler SGD: warmup theo train.warmup_epochs từ factor cấu hình rồi cosine; step theo optimizer updates thành công.
Nếu ngân sách smoke ngắn hơn hoặc bằng warmup, toàn run chỉ warmup (giới hạn warmup bằng số epoch của run).
AMP skipped steps không tăng scheduler; epoch không update nào là lỗi. History LR là LR thực dùng ở update cuối,
không phải next-state. Sau đủ nominal successful updates scheduler next-state đạt min_lr.
Local không tự giảm batch nhóm; Kaggle cần preflight thật để biết batch chung có vừa VRAM.

W&B bắt buộc trong run thật: host/entity/project đọc shared config, một rank0 run xuyên train/final test.
Secrets qua environment/Kaggle Secrets; không tạo connectivity run và không silent offline fallback.
Snapshot host shared hiện bị SDK0.31 từ chối: xem file gửi nhóm để Tuấn xác nhận endpoint đúng trước Kaggle.

Sau training, test tự chạy một lần với best. Standalone eval **một process**:

```powershell
pixi run -e frcnn --locked python -m tools.run_eval --pipeline faster_rcnn --run-dir D:/path/to/run --data-root D:/path/to/same-cache --device cpu
```

Eval dùng saved config/checkpoint, không merge YAML hiện tại, không download weights hoặc init W&B.
`--data-root/--device` là relocation/runtime override tường minh riêng FRCNN. Test bytes/class mapping phải khớp saved run;
không đòi train/val cache hoặc class-table path trên máy train còn tồn tại. Dataset khác phải là đánh giá mới có thiết kế riêng.

## Metric và artifacts

IoU = area(intersection)/area(union). COCO AP tổng hợp precision–recall qua IoU0.50:0.05:0.95;
AP50/AP75 là các ngưỡng riêng, AR100 là average recall với tối đa100 detections/image.
Evaluator bbox, area=all, maxDets=[1,10,100], stats `[0,1,2,8]` ánh xạ bốn keys chung.
Không có prediction nhưng có GT → đo được0; không có GT → null/N/A, không điền0 giả.
COCO categories0..51; model labels được trừ1 khi export.

Score threshold0.001, NMS0.5, max100 được lưu trong config/checkpoint/provenance.
Schema giống YOLO chưa bảo đảm native Ultralytics AP tương đương COCOeval; thống nhất evaluator/policy trước kết luận so sánh.
Không dùng test để tune. README yêu cầu auto-test nên nếu đã nhìn số test thì không gọi blind holdout.
Sau khi chốt config bằng seed42 mới chạy đủ `[42,7,21,123,2026]`; báo từng seed và mean/std với cách tính std rõ.

Mỗi run chứa:

- resolved/effective config, dataset/source identities, history và metrics, inference best checkpoint;
- test_metrics/predictions khi auto-test; curves loss/LR/val metrics;
- fixed positive/negative pool GT xanh/pred đỏ với IDs, class và score; render mọi ảnh được chọn, không chỉ ảnh đẹp;
- run.zip sau final test và W&B finish. Standalone eval có thể thêm sidecars sau zip; repack tường minh nếu cần.

Artifacts không tự là kết quả tốt; phân tích localization/class confusion/small objects/negative false positives sau run thật.
Không bịa good/bad examples hoặc metric khi chưa có run. Tuấn tích hợp geometry/demo/báo cáo và lời giải thích input/output.

## Forge Sweeps

Notebook riêng: `notebooks/frcnn/frcnn_forge_sweeps.ipynb`; notebook03 chung không đổi.
Notebook hiện bàn giao sẵn tuning Bayesian tối đa8trial: source7746b79, seed42,
cap100/warmup3/ES10, batch24/eval32, projectoverride `sandbox`, testpolicy `shared`.
Mỗi phiên agentcount1, mỗi trial dùng cả2GPU. register/agent đều bật: RunAll/Save&RunAll thực sự chạy GPU.
Các bước preflight/pilot đã hoàn tất trước campaign tuning; không thêm overridepatience101 của đối chứng.
Lần đầu giữ `CAMPAIGN_ACTION="new"`. Phiên sau dùng `"continue"`, AddInput bundle+checksum đãexport
(hoặc điền `RESTORE_BUNDLE` path); khôi phục đúngpath/hash/runtime/SweepID. Không tìm đượcID thì dừng,
không tạoSweep thay thế. Source training không đổi so pilot; notebook mới không cần sửa pipeline.
Không chạyhaiagent đồngthời; không tự tạo campaign khác để vượt8attempts. Localcap tính cảfailed,
serverrun_cap giữ8; failed/interrupted không được biến thànhscore0 hoặc tựbù ngoàibudget.
Sau8trial dừng đểreviewwinner vàtốiđa4ablation riêng; notebook không tựchạy đối chứng/finalseeds.
Pilot/tuning giữsandbox, final mới vàoBTL.
Cellcuối export mộtZIP+SHA256 ngay `/kaggle/working`, gom campaign và mọi trial, bỏZIP lồng.
Có thể chạy riêng cellcuối cho campaign đã chạy để tải kết quả, không cần train lại.
Search đề xuất ở `configs/sweeps/faster_rcnn.yaml`: LR/weight decay theo log distribution,
dropout/jitter theo danh sách. Chưa coi search space hoặc ngân sách là tối ưu/đã nghiệm thu.

Workflow: preflight cố định2epochs → pilot baseline ngân sách dài → chốt cap/patience/batch/budget
→ sweep → confirmation → khóa protocol → các seed còn lại. Mỗi campaign mới có thư mục riêng.
Không xếp hạng chung các trial có epoch/cosine budget khác nhau; không dùng test để chọn.

```bash
# Offline prepare: reads/hashes cache, freezes source/config/runtime. No W&B init/training.
# 1 is an example of a finite cap; set the reviewed budget after pilot for tuning.
python -m vn_tsd.pipelines.faster_rcnn.sweep prepare --campaign /kaggle/working/frcnn_sweeps/NAME --data-root /path/to/processed --run-cap 1 --nproc 2 --test-policy shared
# Explicit network operation: register once; saves/reuses sweep_id.json.
python -m vn_tsd.pipelines.faster_rcnn.sweep register --campaign /kaggle/working/frcnn_sweeps/NAME
# Explicit GPU operation; one sequential agent, each trial uses two GPUs.
python -m vn_tsd.pipelines.faster_rcnn.sweep agent --campaign /kaggle/working/frcnn_sweeps/NAME --count 1
```

`--test-policy shared` giữ auto-test theo config nhóm. `skip` là lựa chọn tường minh cho campaign
không chạy test, cần thống nhất nhóm; không thay shared YAML. Notebook mặc định giữ `shared`.
Preflight/pilot dùng grid một bộ tham số baseline; tuning dùng search YAML. Epoch override chỉ nằm
trong campaign, không sửa defaults nhóm. Hai epoch preflight với warmup3 là warmup-only, không phải benchmark.

CLI agent cấp `WANDB_RUN_ID`, `WANDB_SWEEP_ID` và `${args_json_file}` cho launcher.
Launcher không gọi `wandb.init`; nó ghi một config chung rồi chạy torchrun. Pipeline rank0 tạo đúng
một run/trial và đối chiếu sampled config với resolved config trước tải weights. Mọi rank kiểm hash
config/cache; thay source hoặc package versions sau prepare bị từ chối. Jitter0 không tự đổi masterON.

Objective `sweep/best_val_map` là full-precision best validation mAP50–95, log/summary một lần khi
trial hoàn tất. Test và last-val không được dùng làm objective. Không Hyperband trong đợt đầu.
Failed/interrupted không được gán score0; agent có thể nhận trial kế tiếp trong count đã đặt.
Muốn ngừng nhận trial mới hãy Pause/Stop Sweep; Cancel hoặc stop notebook có thể ngắt trial hiện tại.
Linux launcher chuyển stop tới cả process group torchrun, có timeout/kill fallback; NCCL hard failure
và Kaggle session kill vẫn cần kiểm thực tế. V1 không resume training; trial bị ngắt cần runID mới.

Outputs: campaign config/search/source hashes/runtime/cache identity/ID; `trials/<runID>/` giữ pipeline
artifacts, sampled parameters, console log, launch receipt, `run.zip` và checksum SHA256.
Launcher đóng logs rồi repack ZIP, kể cả controlled failure. Không tự upload model artifact/checkpoint;
SaveVersion và tải ZIP/checksum. Khôi phục campaign đúng đường dẫn tuyệt đối khi tiếp tục session,
cùng source/environment/cache; không sửa campaign để trộn các điều kiện thực nghiệm khác nhau.
Nếu đăng ký bị mất phản hồi và chỉ có `registration_started.json`, kiểm Forge và khôi phục ID thật;
không xóa marker rồi đăng ký mù vì có thể tạo trùng Sweep.

Nguồn: [CoreWeave walkthrough](https://docs.coreweave.com/products/wandb/sweeps/walkthrough),
[config/command macros](https://docs.coreweave.com/products/wandb/sweeps/sweep-config-keys).
Local tests dùng W&B giả và CPU/Gloo, không xác nhận Forge private project hoặc Kaggle NCCL.

## Loader cho demo

```python
import torch
from vn_tsd.pipelines.faster_rcnn.model import load_checkpoint
model, metadata = load_checkpoint('.../checkpoints/checkpoint_best.pt', device='cpu')
# image: RGB float32 [3,640,640] in [0,1], already letterboxed with pad114.
with torch.inference_mode():
    output = model([image])[0]
# output['boxes']: xyxy on processed640; output['labels'] - 1: dataset IDs.
names = metadata['dataset']['names']
```

Tuấn letterbox ảnh/video raw và inverse boxes về raw coordinates trước vẽ. Không coi processed coords là raw coords.
Checkpoint lưu state_dict/primitive metadata, strict offline reconstruction FrozenBN eps0, không pickle whole model.

## Nguồn

- [TorchVision Faster R-CNN](https://docs.pytorch.org/vision/0.26/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html), installed source `faster_rcnn.py`, `rpn.py`, `roi_heads.py`, `backbone_utils.py`, `ops/misc.py` cho implementation/loss/FrozenBN.
- [Faster R-CNN](https://arxiv.org/abs/1506.01497), [FPN](https://arxiv.org/abs/1612.03144), [Mask R-CNN/RoIAlign](https://arxiv.org/abs/1703.06870).
- [COCO API evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py), [PyTorch DDP](https://docs.pytorch.org/docs/2.11/generated/torch.nn.parallel.DistributedDataParallel.html).
- README/config/checklist của nhóm; PDF yêu cầu môn trong docs/sources. Không thêm mô hình/thuật toán chưa dùng vào giải thích phương pháp thực tế.
