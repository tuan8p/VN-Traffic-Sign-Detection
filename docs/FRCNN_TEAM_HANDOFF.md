# Những điểm cần phối hợp trước thực nghiệm Faster R-CNN

Ngày audit: 2026-10-19. Đây là ghi chú bàn giao, **chưa phải xác nhận nhóm đã sửa hoặc đồng ý mọi mục**.
Không merge hoặc chỉnh code thành viên khác trong đợt audit này.

## Snapshot đã đọc

| Nhánh | Commit |
|---|---|
| main | `35c37b13489d1bcd0ee3f163dd1f2b1696a5aa70` |
| faster-rcnn local | `5ad07185a474449570b5ae681101f50e27bfafba` |
| yolo | `c1eceb154968156116418a9946e8d6e6b2de40af` |
| preprocessing | `aa0577330ae7676b93386aeb41125587495b33a7` |

Đây là danh tính source ở lần review, không tự khẳng định remote tương lai vẫn như vậy.

## Bình — cache chung và split

Ba lỗi dưới đã được tái hiện bằng fixture tổng hợp của đúng snapshot preprocessing; chưa có cache thật để audit toàn bộ dữ liệu.

| Mức | Vấn đề và source | Việc cần bàn giao |
|---|---|---|
| P1 | [Thiếu test thì gán test = val](https://github.com/tuan8p/VN-Traffic-Sign-Detection/blob/aa0577330ae7676b93386aeb41125587495b33a7/vn_tsd/data/preprocess_offline.py#L538-L540) | Ba split tách biệt hoặc báo thiếu test; không dùng validation làm test. |
| P1 | [Cleanup chỉ xử lý train–test](https://github.com/tuan8p/VN-Traffic-Sign-Detection/blob/aa0577330ae7676b93386aeb41125587495b33a7/vn_tsd/data/preprocess_offline.py#L247-L287); fixture val–test trùng vẫn qua và có thông báo đã xử lý leakage | Kiểm cả train–val, train–test, val–test và lineage augmentation; mô tả giới hạn kiểm tra, không tuyên bố độc lập tuyệt đối từ dHash. |
| P1 | [Tạo output dir nhưng không làm sạch cache cũ](https://github.com/tuan8p/VN-Traffic-Sign-Detection/blob/aa0577330ae7676b93386aeb41125587495b33a7/vn_tsd/data/preprocess_offline.py#L120-L123) | Build cache mới hoặc policy xóa có kiểm soát. Fixture chạy train={a,b} rồi {a}: metadata ghi 1 nhưng disk còn 2 ảnh. |

Cần xác nhận thêm:

- Crop dHash cố định `y=140:500` không đại diện mọi ảnh; kiểm false negative/positive trước khi purge test. Không mặc định crop đó là vùng bỏ padding đúng cho mọi ảnh.
- README yêu cầu offline augmentation, source snapshot chưa hiện thực. Nhóm chốt rõ cache có augmentation hay không; nếu có, chỉ train và ghi parent/source IDs, phép biến đổi/version. Hai model phải dùng cùng cache đã chốt.
- `data.yaml.path` đang là đường dẫn tuyệt đối của máy build. Cần ghi rõ cách mount trên Kaggle và lấy root người dùng truyền làm chuẩn; không phụ thuộc đường dẫn máy Bình.
- Sau purge, thống nhất **final test set** chung, số lượng, lý do loại và manifest. FRCNN không tự dùng các danh sách private `test_clean_files`/`test_leaked_files` để lọc thêm.
- Bàn giao link/version Kaggle cache, commit/script build, `data.yaml`, `meta.json`, class IDs/names, manifest ảnh/nhãn từng split với hash và lineage nếu có. Cây `images/{train,val,test}`, `labels/{train,val,test}`, ảnh 640×640 pad 114, YOLO labels IDs 0..51; không crop classification.

**Ảnh hưởng:** không chặn viết adapter/model/tests bằng fixtures; chặn nghiệm thu cache và chạy thí nghiệm thật. Không sửa cache hoặc chia lại split bên trong FRCNN.

## Tuấn — hạ tầng chung, đóng gói và báo cáo

| Vấn đề | Cách FRCNN xử lý trong ownership | Phần chung còn lại |
|---|---|---|
| [Train CLI](../tools/run_train.py) tạo UUID/timestamp/run directory ở từng process trước `fit()` | Canonicalize tên/path từ rank 0 trong pipeline; notebook được giữ nguyên scaffold theo yêu cầu, chưa truyền shared `--run-dir` | Torchrun từ scaffold vẫn có thể tạo candidate directories rỗng, và CLI in biến path cũ ở mỗi rank. Nên sửa tập trung khi nhóm giao; FRCNN không tự xóa dirs. |
| Batch chung 8/16 chưa được profile 2×T4 | Chia đúng theo world size, không âm thầm đổi batch hoặc gradient accumulation | Hai team chạy preflight rồi cùng quyết định nếu OOM. CUDA/NMS laptop không xác nhận batch này vừa VRAM. |
| W&B host/entity/project lấy từ shared config | Một rank-0 run xuyên train + final test; không silent offline khi required | Xác nhận quyền truy cập/Secrets và host đang dùng trước Kaggle; không gửi API key vào repo. |
| [CI](../.github/workflows/ci.yml) chỉ cài `.[dev]` | Lazy import ML dependencies; tests nặng skip rõ khi thiếu thư viện, thay assertion scaffold riêng FRCNN | Nếu nhóm muốn CI kiểm model thật, thêm job/dependency riêng do owner hạ tầng xử lý. CI tối thiểu xanh không chứng minh model chạy. |
| Eval từ máy khác | Nhánh FRCNN dự kiến nhận `--data-root`/`--device` tường minh; chỉ đổi vị trí/runtime, kiểm dataset identity; không resolve lại config train | Hai option được giới hạn nhánh FRCNN; YOLO giữ hành vi cũ. Nhóm có thể thống nhất hỗ trợ chung sau. |
| Demo cần dùng checkpoint | Loader FRCNN dựng offline, không download; metadata ghi kiến trúc, class map, RGB và processed-coordinate space | Tuấn xử lý ảnh/video gốc, letterbox và inverse coordinate mapping, geometry/UI; không coi tensor prediction là tọa độ ảnh gốc nếu chưa inverse. |

Checkpoint FRCNN v1 phục vụ inference; **chưa hỗ trợ training resume**. Không gọi một run khởi chạy lại là resume.

**Bổ sung từ implementation preflight local:** W&B SDK `0.31.0` xác minh settings (không kết nối mạng) từ chối
`wandb.Settings(base_url="https://forge.coreweave.com")` của shared config với lỗi:
`Invalid Forge server address; use https://api.forge.coreweave.com.`
Tuấn cần xác nhận nhóm dùng instance nào, endpoint SDK và entity/project/quyền tương ứng, rồi sửa shared config theo thống nhất.
FRCNN giữ target đã cấu hình, fail rõ trước pretrained download/train; không tự đổi endpoint hoặc âm thầm tắt W&B.
Các tests orchestration dùng W&B giả, không chứng minh tài khoản/endpoint truy cập được.

**Notebook03 giữ scaffold theo yêu cầu người dùng:** chỉ đổi branch sang faster-rcnn, chưa thực thi bootstrap/notebook.
Scaffold thiếu cài pycocotools; có fetch/checkout -B về remote, cache readiness kiểm lỏng và fallback,
knobs override thông số YAML, connectivity-test W&B riêng và PATH-based python/torchrun. FRCNN kiểm chặt cache/config trong fit,
nhưng không khẳng định các cells scaffold đã tương thích end-to-end Kaggle. Cần chuẩn bị dependency/endpoint/cache và giữ knobs khớp shared trước run thật.
Markdown scaffold còn ghi fit là khung: đó là mô tả cũ được giữ nguyên, không phải trạng thái source FRCNN hiện tại.
Artifacts sẽ có resolved config, history, 4 metrics + provenance, checkpoint best, curves, ví dụ cố định GT/pred và zip.
Notebook/pipeline README bàn giao giải thích backbone/FPN, RPN, RoIAlign, các loss và cách đổi box/class.

Hai PDF giảng viên được đối chiếu bằng trích xuất text toàn bộ, không audit layout:

- `docs/sources/Spec-CV-project-tonghop.pdf`: trang 2 §3.3–3.4 và §5, trang 3 §6.
- `docs/sources/HD-BTL-CV.pdf`: trang 1 §2.1–2.2, trang 2 §2.3–2.5, trang 3 §3, trang 4 phần hiểu/kiểm chứng nội dung.

Nhóm/Tuấn chịu phần geometry/demo, tổng hợp báo cáo/PDF, ghi đóng góp từng người và public repo theo yêu cầu môn.
W&B, 2×T4, 5 seeds và COCO metric cụ thể là contract nội bộ của nhóm; tiếp tục giữ dù PDF không quy định các giá trị đó.

## Quyền — evaluator và tích hợp

- Giữ global train/eval batch, patience, final split/cache/class IDs chung; không tự tăng/giảm riêng một model.
- Snapshot YOLO dùng metric native, `AR100 = null` khi không có COCO AR100. Schema giống nhau chưa chứng minh evaluator tương đương.
- FRCNN dự kiến COCOeval bbox, IoU 0.50:0.05:0.95, area=all, maxDets=[1,10,100], stats `[0,1,2,8]`. Policy model: score threshold 0.001, class-wise NMS 0.5, tối đa 100 detections/image; lưu trong metadata/config.
- Trước so sánh cuối, thống nhất evaluator, score/NMS/max detections và GT/category mapping; hoặc ghi rõ giới hạn, không báo native recall thành COCO AR100. Đây là gate báo cáo so sánh, không chặn implement FRCNN.
- FRCNN chỉ lấy chọn lọc saved-config eval dispatch, metric schema/null handling và logger từ YOLO; không merge cả nhánh hoặc sửa pipeline YOLO.
- Snapshot YOLO còn TODO cho best/log/W&B; import Ultralytics ở top level có thể không phù hợp CI `.[dev]`. Quyền tự kiểm phần đó, FRCNN giữ import nhẹ.

## Điều kiện chuyển sang thực nghiệm

1. Cache final có provenance, ba split hợp lệ và thống nhất cho hai model; xử lý các lỗi P1 hoặc chứng minh không ảnh hưởng cache bàn giao.
2. Kaggle preflight xác nhận dependencies/CUDA NMS, 2×T4 NCCL, batch chung và W&B thật.
3. Chỉ dùng validation chọn config/checkpoint; test tự chạy cuối train theo README. Ghi exposure test, không dùng để tune và không gọi blind holdout nếu đã xem.
4. Sau chốt config mới chạy đủ seeds `[42,7,21,123,2026]`, ghi mean/std và kết quả từng seed. Thiếu run ghi thiếu; không điền kết quả dự kiến.

File này để người dùng gửi nhóm; chưa gửi qua bất kỳ kênh nào, chưa sửa source ngoài phần Faster R-CNN.
