# Experiments

Log các lần train/đánh giá model. Chi tiết đầy đủ từng bước xem `reports/model_report.md`,
`reports/training_report.md`, `reports/evaluation_report.md`.

## Cấu hình model đã chọn

YOLOv8s (pretrained COCO), 129 layer, 11.166.560 tham số, 28,8 GFLOPs, 1 class (`Human`).
Xem lý do chọn ở `docs/architecture.md`.

## Log các lần chạy

| Ngày | Loại | Epoch | imgsz | Batch | Precision | Recall | mAP50 | mAP50-95 | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|
| 2026-07-17 | Smoke test | 2 | 640 | 4 (autobatch) | 0,974 | 0,959 | 0,975 | 0,603 | Chỉ để verify pipeline chạy đúng đầu-cuối; số liệu cao bất thường vì checkpoint COCO đã biết class "person" gần giống `Human` - **không phản ánh chất lượng model thật** |
| 2026-07-17 | Evaluation (trên checkpoint smoke test) | - | 640 | - | 0,967 | 0,973 | 0,968 | 0,597 | Đo trên tập test 606 ảnh, `conf=0.15, iou=0.45`; vẫn dùng checkpoint tạm 2-epoch |
| — | Full training (100 epoch) | 100 | 640 | -1 (autobatch) | | | | | **Chưa chạy** - đã hoãn theo yêu cầu, ước tính ~4,2 giờ (150s/epoch x 100). Xem hướng dẫn chạy ở mục dưới |

## Sự cố & quyết định quan trọng đã xử lý

- **GPU OOM ở imgsz=1280**: quyết định ban đầu giữ nguyên resolution gốc 1280x960 (object nhỏ,
  người ở xa), nhưng RTX 3060 Laptop 6GB VRAM OOM ngay cả với autobatch. Một khi CUDA OOM, cả
  CUDA context của process bị hỏng nên không tự retry batch nhỏ hơn được trong cùng phiên - phải
  khởi động lại kernel. Đã quyết định đổi xuống **imgsz=640** (tổ hợp chuẩn, chắc chắn chạy được).
  Đánh đổi: giảm chi tiết người ở xa/bbox nhỏ, cần theo dõi ảnh hưởng recall.
- **Batch size**: dùng `batch=-1` (autobatch của ultralytics) thay vì số cố định trong
  `configs/training.yaml`, để tránh lặp lại OOM. Ở imgsz=640, autobatch chọn batch=4 trên GPU này.

## Error analysis theo kích thước bbox (checkpoint smoke test)

| Nhóm sqrt(area) bbox | Số lượng | Recall |
|---|---|---|
| 0 - 40 px | 226 | 0,969 |
| 40 - 51,3 px | 226 | 0,987 |
| 51,3 - 74,6 px | 226 | 0,965 |
| 74,6 - 227,7 px | 226 | 0,996 |

Chênh lệch recall nhóm nhỏ nhất vs lớn nhất chỉ ~2,7 điểm % - tín hiệu ban đầu tích cực rằng việc
giảm imgsz 1280→640 chưa ảnh hưởng nặng, nhưng **cần xác nhận lại sau full training** (model mới
học 2 epoch, chưa đủ để kết luận chắc chắn).

## Cách chạy full training khi sẵn sàng

1. Mở `notebooks/05_training.ipynb`, đổi `RUN_FULL_TRAINING = False` → `True` ở mục "4. Full training".
2. Chạy notebook (~4,2 giờ, autobatch). Laptop cần cắm sạc, tránh sleep/hibernate.
3. `outputs/checkpoints/best.pt` sẽ tự động bị ghi đè bởi checkpoint 100-epoch thật.
4. Chạy lại `06_evaluation.ipynb` và `07_inference.ipynb` (không cần sửa code) để có số liệu thật,
   rồi cập nhật lại bảng log ở trên.
