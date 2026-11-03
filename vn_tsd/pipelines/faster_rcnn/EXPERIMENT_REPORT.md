# Faster R-CNN — báo cáo thực nghiệm và bàn giao cuối

Ngày tổng hợp: **03/11/2026**. Người phụ trách: **Trần Anh Tài**.

Đã hoàn thành implementation, preflight, hai pilot, tám trial HPO, năm ablation và bốn seed mới.
Kết quả năm seed (42/7/21/123/2026): **test mAP50–95 53.2887% ± 2.1487 điểm %**.
Checkpoint bàn giao là winner seed 42 `ahgf5kge`, được chọn bằng validation trước khi chạy bốn seed còn lại.

## 1. Mốc xuất phát và phạm vi

- Mốc đối chiếu main: `64bef54d629a41dec8c864f1f73fd7a96cab74af` (`fix wandb`). Đây là mốc main
  sau tích hợp các phần preprocessing/YOLO/Faster R-CNN; chính commit này có một parent, không phải merge commit.
- Các PR trước đó: preprocessing `9ddd240`, Faster R-CNN `0d209a1`, YOLO `0ea73be`.
- Baseline thực nghiệm bên dưới là cấu hình từ scaffold/main, chạy **sau sửa warmup và tích hợp Sweeps**;
  không phải số liệu chạy trực tiếp nguyên trạng commit 64bef54. Legacy 20 epoch không được trộn vào báo cáo này.
- Nhánh Faster R-CNN bổ sung sửa warmup đúng config, bảo vệ artifacts, test-only evaluation,
  chẩn đoán optimizer/AMP và Forge Sweeps. Không sửa preprocessing cache, split hoặc pipeline YOLO.
- Source thực thi: preflight `9ec948e`, hai pilot `7746b79`, HPO/final `a99d294`.
  Pilot và HPO giữ cùng training implementation; các commit sau pilot chủ yếu hoàn thiện notebook/test/tài liệu.
- Notebook hiện hành: [01 — Sweeps](../../../notebooks/frcnn/01_frcnn_forge_sweeps.ipynb),
  [02 — Ablation](../../../notebooks/frcnn/02_frcnn_ablation.ipynb),
  [03 — Final seeds](../../../notebooks/frcnn/03_frcnn_final_seeds.ipynb).

## 2. Protocol chung và cách đọc số liệu

| Thành phần | Thiết lập |
|---|---|
| Model | torchvision Faster R-CNN ResNet-50 FPN, COCO_V1 pretrained, full finetune |
| Nhãn | 52 lớp dataset; predictor 53 gồm background, model label = dataset ID + 1 |
| Custom head | RoI feature 1024 → Linear 512 → ReLU → Dropout → class/bbox predictor |
| Cache | RGB 640 × 640, letterbox pad 114; giữ cùng cache/split/class map qua các run |
| Online | ON; affine jitter theo config; hflip/color jitter/mosaic/mixup OFF |
| Optimizer | SGD, momentum 0.9000 |
| Batch / phần cứng | Global train 24 / eval 32; DDP 2 × Tesla T4; per-rank 12/16; AMP |
| Schedule | Cap 100 epoch, cosine, warmup 3 epoch, start factor 0.0010, min LR 1.0000e-06 |
| Early stopping | Best validation mAP50–95; patience 10, ngoại trừ pilot đối chứng patience 101 |
| Checkpoint / test | Lưu best validation; auto-eval test bằng checkpoint best, không chọn theo test |
| Evaluator | pycocotools 2.0.11 COCOeval bbox; IoU 0.50:0.05:0.95, area all, maxDets 1/10/100 |
| Postprocessing | score threshold 0.0010, NMS 0.5000, tối đa 100 detections/image |
| Runtime ghi nhận | Python 3.13.15; torch 2.11.0+cu128; torchvision 0.26.0+cu128; W&B 0.31.0 |

**Quy ước:** metric trong bảng là phần trăm, trình bày bốn chữ số thập phân.
Std là **sample standard deviation, ddof=1**, ghi bằng điểm phần trăm. Delta cũng là điểm phần trăm.
Tham số hiển thị làm tròn để đọc; **không làm tròn config thực thi/checkpoint**.
Weight decay và min LR dùng dạng khoa học có bốn chữ số thập phân ở mantissa để không biến thành 0.0000.
Nguồn tái lập chính xác: `resolved_config.yaml` trong artifacts; YAML portable trong repo giữ full precision.

## 3. Đường chạy thực nghiệm

| Bước | Run / seed | Mục đích và quyết định |
|---|---|---|
| Preflight | q8r8dglw / 42, 2 epoch | Kiểm Forge Sweep → DDP 2 T4 → train/eval → objective → export; không dùng đánh giá chất lượng |
| Pilot baseline | c5d2ptcg / 42 | Cap 100/ES10; kiểm hội tụ và thời gian |
| Pilot đủ 100 epoch | eby4sb5h / 42 | Cùng baseline, patience 101 để không dừng trong cap 100; khảo sát cải thiện muộn |
| HPO | 8 trial / 42 | Bayesian tìm LR, WD, dropout, jitter; chọn best validation |
| Winner | ahgf5kge / 42 | Trial 6; giữ recipe, không chọn bằng test |
| Ablation | 5 run / 42 | Giữ winner, thay từng mục LR/WD/dropout/jitter/head |
| Final seeds | 7/21/123/2026 | Khóa recipe; train mới, đánh giá ổn định, tổng hợp với 42 |

Tổng **20 training runs trong chuỗi trên**: 1 preflight + 2 pilot + 8 HPO + 5 ablation + 4 final.
Không tính connection-check, legacy 20 epoch, lượt lỗi hoặc lần chạy lại ngoài bộ kết quả đã nghiệm thu.
Chữ `sweep` trong tên pilot chỉ cơ chế khởi chạy; pilot là một cấu hình cố định, không phải search 8 trial.
Preflight ghi project BTL tại thời điểm chạy; các pilot/HPO/ablation dùng sandbox; bốn final dùng BTL.

## 4. Preflight và hai pilot baseline

Baseline: LR 0.0020, WD 5.0000e-04, dropout 0.2000, jitter 0.1000; custom 512.

| Run | Best / stop | Best val mAP50–95 (%) | Test mAP50–95 (%) | History time (giờ) |
|---|---|---:|---:|---:|
| [Preflight q8r8dglw](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/runs/q8r8dglw) | 1 / 2 | 2.1178 | 1.3511 | 0.0884 |
| [Baseline ES10 c5d2ptcg](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/runs/c5d2ptcg) | 56 / 66 | 42.7654 | 48.9982 | 2.5753 |
| [Baseline patience 101 eby4sb5h](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/runs/eby4sb5h) | 87 / 100 | 43.3922 | 49.6389 | 4.0082 |

Pilot ES10 dừng 66 trong khi LR chưa đi hết cosine, nên chạy thêm đối chứng đủ 100 epoch.
Run thứ hai cải thiện 0.6268 điểm % best validation so run thứ nhất. Tuy nhiên hai trajectory khác từ đầu,
không chứng minh riêng việc tắt ES gây ra mức tăng đó; mô phỏng ES10 trên trajectory thứ hai vẫn giữ best 87, dừng 97.
Quyết định: giữ cap 100/warmup 3/ES10, search hữu hạn 8 trial, không mặc định bỏ ES cho mọi trial.
History time chỉ gồm các epoch, không phải toàn bộ wall-clock startup/test/export hoặc benchmark tốc độ.

## 5. Bayesian Sweep — tám bộ hyperparameters

Search space: LR log-uniform [0.0005, 0.0100]; WD log-uniform [1.0000e-05, 1.0000e-03];
dropout {0.0000, 0.2000, 0.4000}; jitter {0.0000, 0.0500, 0.1000, 0.1500}.
Mỗi trial thử **một bộ bốn tham số đồng thời**, giữ kiến trúc custom 512 và protocol chung.
Objective `sweep/best_val_map`, maximize; không dùng Hyperband pruning trong campaign này.
[Sweep upxyjxyl](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/sweeps/upxyjxyl).

**Lưu ý**: run [ahgf5kge](https://forge.coreweave.com/wandb/HCMUT_IPCV/BTL/runs/ahgf5kge) trên Forge đã được move từ project [sandbox](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox) sang project [BTL](https://forge.coreweave.com/wandb/HCMUT_IPCV/BTL) vì đây là best run của seed 42, đem sang BTL để group chung với 4 seeds còn lại làm final result. Vậy nên, trong [Sweep upxyjxyl](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/sweeps/upxyjxyl) hiện chỉ đang display 7 runs còn lại.

| Trial | Run ID | LR | WD | Dropout | Jitter | Best val mAP50–95 (%) | Best / stop |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | [dpuw3rbm](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/runs/dpuw3rbm) | 0.0012 | 4.4160e-04 | 0.0000 | 0.0500 | 37.2970 | 47 / 57 |
| 2 | [9sk2wnb6](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/runs/9sk2wnb6) | 0.0084 | 3.0407e-05 | 0.4000 | 0.1000 | 48.6665 | 29 / 39 |
| 3 | [4fnjojd1](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/runs/4fnjojd1) | 0.0050 | 2.4523e-05 | 0.4000 | 0.1000 | 46.0870 | 29 / 39 |
| 4 | [5u0jju9u](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/runs/5u0jju9u) | 0.0097 | 4.5614e-05 | 0.4000 | 0.1500 | 47.9487 | 29 / 39 |
| 5 | [l4fegpdz](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/runs/l4fegpdz) | 0.0097 | 1.5949e-05 | 0.4000 | 0.0500 | 48.2753 | 29 / 39 |
| 6 | [ahgf5kge](https://forge.coreweave.com/wandb/HCMUT_IPCV/BTL/runs/ahgf5kge) | 0.0096 | 1.4757e-05 | 0.4000 | 0.1000 | 49.0528 | 49 / 59 |
| 7 | [i3lu2i4b](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/runs/i3lu2i4b) | 0.0091 | 1.1345e-05 | 0.2000 | 0.1500 | 48.2207 | 26 / 36 |
| 8 | [9n1l8ngw](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/runs/9n1l8ngw) | 0.0098 | 1.2009e-05 | 0.0000 | 0.0000 | 40.7800 | 32 / 42 |

Winner **trial 6 / ahgf5kge**: LR 0.0096, WD 1.4757e-05, dropout 0.4000, jitter 0.1000.
Best validation 49.0528% tại epoch 49, stop 59; baseline 42.7654%, delta **+6.2875 điểm %**.
Jitter winner bằng baseline; chỉ LR/WD/dropout đổi giữa baseline và winner.
Đây là winner trong tám trial của search space này, không khẳng định tối ưu toàn cục.
Tám trial không đủ xác định mọi tương tác hoặc đóng góp nhân quả của từng tham số.

## 6. Năm ablation sau HPO

Mỗi run seed 42 train mới từ COCO pretrained, giữ winner và chỉ thay mục được ghi.
LR/WD/dropout trả baseline; jitter 0 bỏ affine jitter nhưng master online vẫn ON;
standard thay custom predictor, các field dropout/hidden của custom không có tác dụng với standard.

| Đối chứng | Thay riêng | Run ID | Best / stop | Best val mAP50–95 (%) | Delta vs winner (điểm %) | Test mAP50–95 (%) |
|---|---|---|---|---:|---:|---:|
| lr | LR → 0.0020 | [e4c2164d](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/runs/e4c2164d) | 78 / 88 | 44.8842 | -4.1686 | 50.8664 |
| weight_decay | WD → 5.0000e-04 | [0c21d69d](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/runs/0c21d69d) | 23 / 33 | 47.2168 | -1.8361 | 52.0370 |
| dropout | Dropout → 0.2000 | [b793131b](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/runs/b793131b) | 39 / 49 | 48.5910 | -0.4618 | 54.4504 |
| jitter | Jitter → 0.0000 | [3d3bfb86](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/runs/3d3bfb86) | 22 / 32 | 42.0392 | -7.0136 | 46.3384 |
| standard | Head → standard | [b27d25aa](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/runs/b27d25aa) | 23 / 33 | 46.9988 | -2.0540 | 50.9730 |

Cả năm đối chứng thấp hơn winner trên validation của seed 42, hỗ trợ quyết định giữ recipe.
Dropout chênh 0.4618 điểm %, chưa chứng minh ổn định qua nhiều seed. Custom-vs-standard là đối chứng
trên recipe tuned cho custom, không phải hai kiến trúc được tuning ngang ngân sách.
Không cộng các delta hoặc gọi chúng là đóng góp độc lập. Jitter 0.1000 được giữ do kết quả trong recipe này,
không dùng legacy 20 epoch để kết luận cường độ tối ưu.

## 7. Khóa cấu hình và chạy bốn seed còn lại

- [Baseline YAML](../../../configs/pipelines/faster_rcnn_baseline.yaml): giữ nguyên bộ trước Sweep.
- [Default winner YAML](../../../configs/pipelines/faster_rcnn.yaml): recipe tuned, full precision.
- [Protocol record](../../../configs/experiments/faster_rcnn_final_protocol.json): frcnn_final_v1;
  hồ sơ chủ động tạo, tham chiếu config/source/data, không phải runtime-generated receipt.
- Notebook final đọc recipe từ bundle e2d80e82 đã pin SHA256, không load checkpoint winner.
- Hai session Kaggle: [7, 21] và [123, 2026], mỗi session train nối tiếp; mỗi seed có run/checkpoint/ZIP riêng.
- Reuse 42 ahgf5kge theo kế hoạch. Không chọn lại seed có test cao nhất, không loại seed 21 vì điểm thấp.

## 8. Bảng kết quả cuối — baseline, năm seed và mean/std

Tất cả chỉ số bên dưới tính theo %. Baseline hai hàng đầu đều seed 42, không tính vào mean/std winner.
Validation là tại checkpoint best; test đánh giá chính checkpoint đó. Best/stop có thể khác nhau do ES.

| Cấu hình / seed | Run ID | Best / stop | Val AP50–95 | Val AP50 | Val AP75 | Val AR100 | Test AP50–95 | Test AP50 | Test AP75 | Test AR100 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline ES10 / 42 | [c5d2ptcg](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/runs/c5d2ptcg) | 56 / 66 | 42.7654 | 59.1895 | 50.2300 | 62.5741 | 48.9982 | 67.2786 | 57.9701 | 63.5696 |
| Baseline no-ES-in-cap / 42 | [eby4sb5h](https://forge.coreweave.com/wandb/HCMUT_IPCV/sandbox/runs/eby4sb5h) | 87 / 100 | 43.3922 | 59.5754 | 51.9599 | 62.8474 | 49.6389 | 67.6342 | 59.7013 | 64.1438 |
| Winner / 42 | [ahgf5kge](https://forge.coreweave.com/wandb/HCMUT_IPCV/BTL/runs/ahgf5kge) | 49 / 59 | 49.0528 | 65.2268 | 59.2304 | 65.2143 | 55.1916 | 74.0401 | 68.5586 | 66.1100 |
| Winner / 7 | [e7dcff6b](https://forge.coreweave.com/wandb/HCMUT_IPCV/BTL/runs/e7dcff6b) | 47 / 57 | 48.7992 | 64.6382 | 59.2502 | 67.8334 | 54.5724 | 72.3874 | 68.5397 | 69.0470 |
| Winner / 21 | [f6e787cc](https://forge.coreweave.com/wandb/HCMUT_IPCV/BTL/runs/f6e787cc) | 21 / 31 | 48.7134 | 66.6783 | 60.0590 | 68.3843 | 49.8500 | 70.4040 | 60.3785 | 64.9016 |
| Winner / 123 | [cd051bc1](https://forge.coreweave.com/wandb/HCMUT_IPCV/BTL/runs/cd051bc1) | 28 / 38 | 49.4229 | 67.7325 | 60.7489 | 65.6082 | 54.2340 | 72.9432 | 66.7962 | 69.1081 |
| Winner / 2026 | [2273ba21](https://forge.coreweave.com/wandb/HCMUT_IPCV/BTL/runs/2273ba21) | 37 / 47 | 49.8177 | 66.5638 | 59.6363 | 66.5262 | 52.5953 | 70.8543 | 65.6026 | 65.8296 |
| **Mean ± std, 5 seed** | — | — | 49.1612 ± 0.4590 | 66.1679 ± 1.2340 | 59.7849 ± 0.6364 | 66.7133 ± 1.3739 | 53.2887 ± 2.1487 | 72.1258 ± 1.4985 | 65.9751 ± 3.3685 | 66.9993 ± 1.9493 |
| **Mean ± std, 4 seed mới** | — | — | 49.1883 ± 0.5254 | 66.4032 ± 1.2889 | 59.9236 ± 0.6417 | 67.0880 ± 1.2572 | 52.8129 ± 2.1557 | 71.6472 ± 1.2113 | 65.3292 ± 3.5139 | 67.2216 ± 2.1764 |

Mean/std được tính từ số liệu full precision, không tính từ bảng đã làm tròn.
Validation mAP50–95 giữa năm seed khá sát nhau (std 0.4590 điểm %); test dao động lớn hơn (std 2.1487 điểm %).
Seed 21 thấp hơn trên test nhưng artifact/config/history/checkpoint đều hợp lệ; không có cơ sở loại run hoặc train lại theo điểm.
Baseline → winner seed 42 là so sánh matched single-seed; mean năm winner không tự chứng minh gain đa seed so baseline/standard.

## 9. Nghiệm thu, giới hạn và chạy song song

- Bốn ZIP final và bundle winner: checksum/CRC, config/source/runtime/cache/evaluator consistency qua.
- World 2/AMP/global 24/eval 32/fresh COCO, ES10 và cosine theo successful updates khớp history.
- W&B run IDs/test summary khớp metrics; checkpoint SHA256/metadata/finite weights/strict CPU load qua cả năm seed.
- Không thực hiện tái tính AP hoặc real-image inference độc lập trong lần review artifacts.
- Artifacts không xác nhận topology shared CPU/dedicated GPU. CPU contention nếu có chủ yếu ảnh hưởng tốc độ;
  code dừng theo epoch/metric và schedule theo update, không theo elapsed time. Không có bằng chứng phải rerun do hai session song song.
- Seed đã đặt cho sampler/loader/model nhưng không bật đầy đủ deterministic algorithms; không hứa bitwise reproducibility.
  [PyTorch 2.11 reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html).
- Seed 42 từng được dùng chọn hyperparameters; bảng năm seed là tổng hợp theo kế hoạch, không phải năm seed untouched sau HPO.
  Bảng bốn seed mới được báo riêng để phân biệt. Rerun 42 không tự loại bỏ ảnh hưởng của HPO-selection seed.
- Dữ liệu có cảnh/frame gần nhau giữa split; test đã được quan sát trong tuning dù không dùng chọn cấu hình.
  Không tuyên bố blind holdout hoặc khả năng tổng quát trên cảnh mới từ các kết quả này.
- Native metrics của YOLO và COCOeval FRCNN cần thống nhất evaluator/policy trước kết luận hơn-kém trực tiếp.
- Standard ablation từng lỗi export symlink sau train; bản whole-output đã kiểm internal hashes/strict load.
  Thiếu checksum nguồn cho outer ZIP standard được ghi nhận, không coi checksum local là kiểm nguồn độc lập.

## 10. Nguồn bằng chứng

Số liệu lấy từ artifacts đã review: preflight q8r8dglw; pilot c5d2ptcg/eby4sb5h;
Sweep upxyjxyl tám trials; năm ablations; final 42/7/21/123/2026.
Resolved configs, metrics, histories, source/dataset identities và checkpoint hashes là bằng chứng thực thi.
Các receipt local trong `.agents/artifacts/notebooks/` là bản kiểm hỗ trợ, không đưa `.agents` lên Git.
Local suite 114 passed xác nhận các scope code đã test, không thay thế kết quả training thực.
Không có live private-Forge query trong lần tổng hợp báo cáo này; link Forge dành cho thành viên có quyền truy cập.
