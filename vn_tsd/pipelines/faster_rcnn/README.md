# Faster R-CNN

**Kết quả và bàn giao cuối (03/11/2026):** xem [báo cáo thực nghiệm](EXPERIMENT_REPORT.md).
Báo cáo gồm mốc main `64bef54`, preflight, hai pilot, tám trial tuning, năm ablation,
baseline/năm seed/mean ± std và thông tin `run.zip` winner seed 42. Chuỗi thực nghiệm đã hoàn tất;
các mục bên dưới mô tả implementation và cách vận hành, không thay thế báo cáo kết quả.

Pipeline nhận **ảnh processed RGB 640 × 640 cùng YOLO labels**, trả boxes/class/scores trên tọa độ ảnh processed.
Không tạo cache offline, đổi split hoặc crop từng biển thành classification. Online augmentation chỉ áp vào train.
Implementation không đồng nghĩa đã có kết quả huấn luyện; random-weight smoke chỉ kiểm đường chạy.

## Phương pháp thực sự dùng

ResNet-50 trích đặc trưng; FPN ghép thông tin nhiều mức phân giải, giúp biểu diễn vật thể có nhiều kích thước.
RPN đề xuất vùng dựa trên anchors, dự đoán objectness và độ lệch box. RoIAlign lấy đặc trưng vùng với nội suy,
tránh lượng tử hóa vị trí như pooling đơn giản. RoI head dự đoán class và tinh chỉnh box; score filtering,
class-wise NMS và giới hạn số detection tạo output cuối. Giữ anchors/RPN/RoI sampling mặc định của TorchVision v1 baseline;
chưa có bằng chứng đây là lựa chọn tối ưu cho biển báo nhỏ.

Pretrained COCO V1 được nạp với 91 outputs rồi thay head thành **53**: background 0 + dataset IDs 0..51 được cộng 1.
Mọi learnable parameter train ngay từ epoch 1; FrozenBatchNorm giữ thống kê pretrained và eps 0, không có lịch freeze/unfreeze hai stage.
Đây là full finetune parameter, không phải cập nhật lại running statistics của BatchNorm.

### Custom head và baseline chuẩn

Theo góp ý thêm layers của Tuấn, YAML hiện chọn `model.head: custom_mlp`. Giữ nguyên pretrained backbone/FPN/RPN và
hai FC layers của `roi_heads.box_head`; thay riêng predictor sau feature vector 1024 bằng:

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
Đổi `model.head` sang `standard` giữ predictor TorchVision 53 của baseline, không thêm projection/Dropout. Ví dụ override ablation:

```yaml
model:
  head: standard  # custom_mlp để dùng head thêm layers
```

`head_hidden_dim` (mặc định 512) và `head_dropout` (0.2) chỉ có tác dụng với custom_mlp. Khi so sánh head chuẩn/custom,
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
Cache đã letterbox 640, vì vậy flag letterbox/normalize là yêu cầu đã được thỏa, không phải hai augmentation thêm.
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
Ảnh phải decode được và 640 × 640, label file phải tồn tại (rỗng là negative hợp lệ). Mỗi dòng là integer class 0..51 + normalized cx,cy,w,h.
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
model checkpoint thật, inference 640, backward 128, fixtures và orchestration hai CPU/Gloo ranks. Tiny model chỉ kiểm điều phối;
không chứng minh NCCL 2 × T4 hoặc chất lượng detection. Không test nào tải data/weights hoặc gọi W&B thật.

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

CLI smoke này vẫn forward/train ở 640 và có thể chậm/nặng trên laptop. Không chạy mặc định trong setup hoặc coi nó là full-run result.
Batch smoke khác batch nhóm được ghi trong resolved/effective config; không đưa số smoke vào bảng so sánh.

## Kaggle full training và eval

Notebook [03](../../../notebooks/03_train_faster_rcnn.ipynb) giữ cấu trúc scaffold của Tuấn;
chỉ đồng bộ ba giá trị mặc định sau HPO: LR `0.009606472124890571`, weight decay `1.4756984677083597e-05`
và dropout `0.4`, cùng ghi chú recipe winner `ahgf5kge`. Không thêm cell hoặc đổi bootstrap/launcher.
Notebook chạy một run mới, mặc định seed 42, custom 512 / online ON / jitter 0.1 / cap 100 / warmup 3 / ES10 / global 24 / eval 32;
không phải Sweep hay chạy tiếp checkpoint winner. Chạy lại seed42 không tự thay kết quả gốc trong bảng năm seed.
`REPO_BRANCH="main"` giữ nguyên scaffold: cần merge PR vào main trước khi bootstrap lấy các thay đổi mới;
nếu kiểm trước merge thì chọn nhánh `faster-rcnn` đã push. Bootstrap fetch rồi checkout -B về remote;
không chạy cell này trên local đang có code chưa push.
Scaffold cài `.[detect]` (đã gồm pycocotools) và W&B, đọc Secret rồi chỉ rank 0 tạo training run;
không có connectivity-test run riêng. Các biến override có thể ghi đè YAML, cần giữ đúng recipe đã chốt.
Tracking project lấy từ shared config (BTL); nếu chạy smoke 2 epoch, dùng override tạm sang sandbox,
không ghi cấu hình smoke vào defaults final hoặc đưa số smoke vào bảng kết quả.
Đã kiểm local cấu hình resolve và launcher; chưa xác nhận phiên bản notebook03 đồng bộ này chạy Kaggle end-to-end.
Chưa hỗ trợ chạy lại vào run-dir đã có resolved_config/checkpoint; chọn run mới. V1 **không training resume**.

Global batch train/eval và patience đọc shared config; per-rank batch chia theo WORLD_SIZE thực.
Training DistributedSampler có thể padding sample khi N không chia hết world size (được log), tail batch không drop.
Val/test shard không pad/drop; unwrapped model forward, gather predictions một lần sau shard và rank 0 COCOeval.
Rank 0 chọn best strict val mAP50-95 tăng; score 0 vẫn lưu checkpoint đầu, tie không overwrite, stop đúng patience.

Scheduler SGD: warmup theo train.warmup_epochs từ factor cấu hình rồi cosine; step theo optimizer updates thành công.
Nếu ngân sách smoke ngắn hơn hoặc bằng warmup, toàn run chỉ warmup (giới hạn warmup bằng số epoch của run).
AMP skipped steps không tăng scheduler; epoch không update nào là lỗi. History LR là LR thực dùng ở update cuối,
không phải next-state. Sau đủ nominal successful updates scheduler next-state đạt min_lr.
Local không tự giảm batch nhóm; Kaggle cần preflight thật để biết batch chung có vừa VRAM.

W&B bắt buộc trong run thật: host/entity/project đọc shared config, một rank 0 run xuyên train/final test.
Secrets qua environment/Kaggle Secrets; không tạo connectivity run và không silent offline fallback.
Endpoint hiện tại lấy từ shared config: API `https://api.forge.coreweave.com`, UI `https://forge.coreweave.com/wandb`;
đã dùng trong các run thực nghiệm, phiên Kaggle mới vẫn cần Secret và quyền ghi project.

Sau training, test tự chạy một lần với best. Standalone eval **một process**:

```powershell
pixi run -e frcnn --locked python -m tools.run_eval --pipeline faster_rcnn --run-dir D:/path/to/run --data-root D:/path/to/same-cache --device cpu
```

Eval dùng saved config/checkpoint, không merge YAML hiện tại, không download weights hoặc init W&B.
`--data-root/--device` là relocation/runtime override tường minh riêng FRCNN. Test bytes/class mapping phải khớp saved run;
không đòi train/val cache hoặc class-table path trên máy train còn tồn tại. Dataset khác phải là đánh giá mới có thiết kế riêng.

## Metric và artifacts

IoU = area(intersection)/area(union). COCO AP tổng hợp precision–recall qua IoU 0.50:0.05:0.95;
AP50/AP75 là các ngưỡng riêng, AR100 là average recall với tối đa 100 detections/image.
Evaluator bbox, area=all, maxDets=[1, 10, 100], stats `[0,1,2,8]` ánh xạ bốn keys chung.
Không có prediction nhưng có GT → đo được 0; không có GT → null/N/A, không điền 0 giả.
COCO categories 0..51; model labels được trừ 1 khi export.

Score threshold 0.001, NMS 0.5, max 100 được lưu trong config/checkpoint/provenance.
Schema giống YOLO chưa bảo đảm native Ultralytics AP tương đương COCOeval; thống nhất evaluator/policy trước kết luận so sánh.
Không dùng test để tune. README yêu cầu auto-test nên nếu đã nhìn số test thì không gọi blind holdout.
Sau khi chốt config bằng seed 42 mới chạy đủ `[42,7,21,123,2026]`; báo từng seed và mean/std với cách tính std rõ.

Mỗi run chứa:

- resolved/effective config, dataset/source identities, history và metrics, inference best checkpoint;
- test_metrics/predictions khi auto-test; curves loss/LR/val metrics;
- fixed positive/negative pool GT xanh/pred đỏ với IDs, class và score; render mọi ảnh được chọn, không chỉ ảnh đẹp;
- run.zip sau final test và W&B finish. Standalone eval có thể thêm sidecars sau zip; repack tường minh nếu cần.

Artifacts không tự là kết quả tốt; phân tích localization/class confusion/small objects/negative false positives sau run thật.
Không bịa good/bad examples hoặc metric khi chưa có run. Tuấn tích hợp geometry/demo/báo cáo và lời giải thích input/output.

## Forge Sweeps

Notebook riêng: [01_frcnn_forge_sweeps.ipynb](../../../notebooks/frcnn/01_frcnn_forge_sweeps.ipynb); notebook 03 chung không đổi.
Notebook hiện bàn giao sẵn tuning Bayesian tối đa 8 trial: source 7746b79, seed 42,
cap 100/warmup 3/ES10, batch 24/eval 32, project override `sandbox`, test policy `shared`.
Mỗi phiên agent count 1, mỗi trial dùng cả 2 GPU. register/agent đều bật: RunAll/Save&RunAll thực sự chạy GPU.
Các bước preflight/pilot đã hoàn tất trước campaign tuning; không thêm override patience 101 của đối chứng.
Lần đầu giữ `CAMPAIGN_ACTION="new"`. Phiên sau dùng `"continue"`, Add Input bundle + checksum đã export
(hoặc điền `RESTORE_BUNDLE` path); khôi phục đúng path/hash/runtime/SweepID. Không tìm được ID thì dừng,
không tạo Sweep thay thế. Source training không đổi so pilot; notebook mới không cần sửa pipeline.
Không chạy hai agent đồng thời; không tự tạo campaign khác để vượt 8 attempts. Local cap tính cả failed,
server run_cap giữ 8; failed/interrupted không được biến thành score 0 hoặc tự bù ngoài budget.
Sau 8 trial dừng để review winner và tối đa 4 ablation riêng; notebook không tự chạy đối chứng/final seeds.
Pilot/tuning giữ sandbox, final mới vào BTL.
Cell cuối export một ZIP + SHA256 ngay `/kaggle/working`, gom campaign và mọi trial, bỏ ZIP lồng.
Có thể chạy riêng cell cuối cho campaign đã chạy để tải kết quả, không cần train lại.
Search đề xuất ở `configs/sweeps/faster_rcnn.yaml`: LR/weight decay theo log distribution,
dropout/jitter theo danh sách. Chưa coi search space hoặc ngân sách là tối ưu/đã nghiệm thu.

Workflow: preflight cố định 2 epochs → pilot baseline ngân sách dài → chốt cap/patience/batch/budget
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
trong campaign, không sửa defaults nhóm. Hai epoch preflight với warmup 3 là warmup-only, không phải benchmark.

CLI agent cấp `WANDB_RUN_ID`, `WANDB_SWEEP_ID` và `${args_json_file}` cho launcher.
Launcher không gọi `wandb.init`; nó ghi một config chung rồi chạy torchrun. Pipeline rank 0 tạo đúng
một run/trial và đối chiếu sampled config với resolved config trước tải weights. Mọi rank kiểm hash
config/cache; thay source hoặc package versions sau prepare bị từ chối. Jitter 0 không tự đổi master ON.

Objective `sweep/best_val_map` là full-precision best validation mAP50–95, log/summary một lần khi
trial hoàn tất. Test và last-val không được dùng làm objective. Không Hyperband trong đợt đầu.
Failed/interrupted không được gán score 0; agent có thể nhận trial kế tiếp trong count đã đặt.
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

## Ablation sau Sweep seed 42

Notebook [02_frcnn_ablation.ipynb](../../../notebooks/frcnn/02_frcnn_ablation.ipynb) chạy **một đối chứng mỗi phiên**.
Năm lựa chọn `ABLATION`: `lr`, `weight_decay`, `dropout`, `jitter`, `standard`;
mỗi bản Kaggle chọn một mục và có output riêng.
Mỗi phiên dùng T4 × 2, tạo run Forge độc lập trong sandbox, group `frcnn_ablation_s42_v1`;
không tạo thêm trial trong Sweep tuning. Có thể chạy song song ở các phiên GPU độc lập.

Add Input cache cũ và Output tuning cuối có `...-artifacts-e2d80e82.zip` + `.zip.sha256`.
`REFERENCE_BUNDLE` tự tìm đúng tên bundle hoặc nhận Copy file path khi có nhiều Input.
Notebook pin SHA256 bundle, source a99d294, đối chiếu runtime/cache và đọc config winner
`ahgf5kge`; không load hoặc extract checkpoint winner. Giữ seed 42/cap 100/warmup 3/ES10,
batch 24/eval 32 và fresh COCO pretrained. Mỗi run giữ recipe winner, chỉ đổi một mục:
LR 0.002, WD 0.0005, dropout 0.2, jitter 0.0, hoặc head=`standard`.
Ba mục đầu trả về baseline. Jitter 0 tắt affine jitter, master online vẫn ON;
baseline jitter thực tế là 0.1 nên đây là đối chứng bỏ thành phần, không phải trả về baseline.
Ở standard, raw fields head_hidden_dim/head_dropout giữ nguyên nhưng không áp dụng vào predictor.
Đây là so sánh thay head dưới recipe đã chọn cho custom; chưa phải standard đã được tuning riêng.
Các đối chứng seed 42 đo tác dụng có điều kiện, không tách toàn bộ tương tác hoặc chứng minh
ổn định qua nhiều seed. Dùng validation, không test, để quyết định cấu hình cuối.

`START_TRAIN=True` đã đặt sẵn; Save & Run All thực sự train. Mỗi phiên export ZIP/checksum
ra Output gốc trong `finally`, kể cả controlled failure. Khi cần khôi phục output của chính
đối chứng, Add Input ZIP/checksum đó và đặt `RESTORE_OUTPUT`; completed run được kiểm rồi
bỏ qua, failed/interrupted không tự retry. Hard session kill không bảo đảm `finally` chạy.
Giữ Save Version outputs. V1 không resume training hoặc tự upload checkpoint Artifact lên Forge.
Exporter bỏ qua symbolic links (W&B log aliases/latest-run), không dereference chúng;
file thường và checkpoint vẫn được đóng gói, danh sách link bỏ qua nằm trong export_receipt.json.
Nếu notebook cũ báo `Symlink output rejected` sau train, lỗi nằm ở export sau training;
bản notebook chính hiện đã bỏ qua links. Không cần thêm notebook cứu riêng.
Nếu session cũ còn sống, có thể thay riêng hàm export rồi gọi lại để đóng gói kết quả đã có;
không khởi động training chỉ để sửa ZIP. Nếu chạy một thí nghiệm mới, import bản chính đã sửa.

Source training pin đã có trên remote, nên notebook mới có thể chạy khi được import mà không
cần thay REPO_REF sang commit chứa notebook. Local tests dùng cache/worker giả, không xác nhận
Kaggle CUDA/NCCL, quyền project, hoặc khả năng chạy đồng thời nhiều phiên GPU.

## Final seeds sau HPO/ablation — khóa ngày 2026-11-02

`configs/pipelines/faster_rcnn_baseline.yaml` giữ nguyên cấu hình trước Sweep.
`configs/pipelines/faster_rcnn.yaml` là cấu hình winner mặc định dùng tiếp.
`configs/experiments/faster_rcnn_final_protocol.json` lưu protocol, đường dẫn/hash cấu hình,
source/reference identity và giới hạn; không nhân bản các siêu tham số đã có trong YAML.
Winner seed 42 `ahgf5kge`: custom 512/dropout 0.4/LR 0.009606472124890571/
WD 0.000014756984677083597/jitter 0.1, cap 100/warmup 3/ES10/global 24/eval 32.
Dùng lại seed 42 hiện có trong sandbox để tổng hợp năm seed. Có thể chạy lại 42 để kiểm lặp;
run lặp được đánh dấu, không tự thay run gốc hay cộng thành seed thứ sáu.

Import [03_frcnn_final_seeds.ipynb](../../../notebooks/frcnn/03_frcnn_final_seeds.ipynb). Nhận một seed `SEED=21` hoặc danh sách có thứ tự
`SEED=[21,7]`, từ một đến năm seed khác nhau trong `[42,7,21,123,2026]`.
Không dùng set `{21,7}` vì không giữ thứ tự. Hai phiên song song có thể đặt
`SEED=[7,21]` và `SEED=[123,2026]`; mỗi phiên chạy nối tiếp, mỗi run dùng hai T4.
Attach cùng cache và Output tuning cuối có ZIP e2d80e82 + checksum như ablation; T4 × 2,
Internet ON/Secret WANDB_API_KEY. `START_TRAIN=True`; Save & Run All train từng run độc lập trong
**BTL**, group `frcnn_final_v1`. Không cần output seed trước; dùng cùng reference cho cả bốn seed.
Source pin a99d294 đã publish; notebook lấy recipe trực tiếp từ bundle đã pin hash,
không đọc YAML/JSON mới ở checkout cũ. Không đổi source pin sang commit chứa cấu hình mới trong cùng protocol.
Notebook 03 giữ scaffold; ba defaults LR/WD/dropout đã đồng bộ winner, các biến override vẫn có thể ghi đè YAML.
Notebook kiểm source/runtime/cache và đọc JSON/YAML tham chiếu, không load checkpoint winner.

ZIP/checksum xuất ở Output gốc (bỏ qua W&B symlinks); lưu local tại
`.agents/artifacts/notebooks/frcnn_final_seeds/seed_<seed>/` rồi gửi review.
Mỗi seed export ngay một cặp ZIP/checksum. `RESTORE_OUTPUT` nhận ZIP của đúng seed khi chạy đơn,
hoặc dict `{7: "...zip", 21: "...zip"}` khi chạy nhiều seed, để kiểm completed và skip.
Failed/dở dừng hàng đợi, không tự retry hoặc resume training. Giữ Save Version outputs;
hard-kill không bảo đảm finally chạy. Chọn số seed sao cho đủ giới hạn thời gian phiên Kaggle.
Review đủ bốn seed mới tổng hợp cùng seed 42, mean và sample std ddof=1; không chọn/bỏ seed theo score.
Các metric test chỉ report; split cảnh/frame và test exposure hiện tại vẫn phải ghi trong báo cáo.
Mean/std của winner không thay cho paired multi-seed benchmark với baseline hoặc standard.

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
Checkpoint lưu state_dict/primitive metadata, strict offline reconstruction FrozenBN eps 0, không pickle whole model.

## Nguồn

- [TorchVision Faster R-CNN](https://docs.pytorch.org/vision/0.26/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html), installed source `faster_rcnn.py`, `rpn.py`, `roi_heads.py`, `backbone_utils.py`, `ops/misc.py` cho implementation/loss/FrozenBN.
- [Faster R-CNN](https://arxiv.org/abs/1506.01497), [FPN](https://arxiv.org/abs/1612.03144), [Mask R-CNN/RoIAlign](https://arxiv.org/abs/1703.06870).
- [COCO API evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py), [PyTorch DDP](https://docs.pytorch.org/docs/2.11/generated/torch.nn.parallel.DistributedDataParallel.html).
- README/config/checklist của nhóm; PDF yêu cầu môn trong docs/sources. Không thêm mô hình/thuật toán chưa dùng vào giải thích phương pháp thực tế.
