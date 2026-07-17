# Báo cáo khởi tạo model (Model Report)

Sinh từ `notebooks/04_model.ipynb` (chạy ngày 2026-07-17, kernel `thermal_env`).
Theo cấu hình đã chốt trong `configs/model.yaml` và bài toán ở `docs/architecture.md`.

## 1. Kiến trúc đã chọn

| Hạng mục | Giá trị |
|---|---|
| Model | YOLOv8s (pretrained COCO) |
| Số layer | 129 |
| Tham số | 11.166.560 |
| GFLOPs | 28,8 |
| Input size | 1280x960 (giữ nguyên resolution gốc, không downscale) |
| Số class (sau fine-tune) | 1 (`Human`) |

## 2. Kết quả kiểm tra

- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU, model load và chạy được trên `cuda:0`
- **Smoke test forward pass**: chạy `predict` trên 1 ảnh thật từ `data/processed/images/train` (đã qua denoise+CLAHE) - không lỗi kích thước/kênh ảnh, phát hiện 2 box (checkpoint COCO gốc, **chưa fine-tune** nên không phản ánh độ chính xác thật trên bài toán này)
- **Đối chiếu cấu hình**: `nc`/`names` trong `data/processed/data.yaml` khớp đúng `num_classes`/`class_names` trong `configs/model.yaml` - không lệch số class trước khi sang training

## 3. Code liên quan

- `src/models/yolov8.py` - hàm `build_yolov8()`, cache checkpoint pretrained tại `outputs/checkpoints/` (tránh ultralytics tự tải `.pt` vào thư mục hiện tại mỗi lần chạy khác nhau)
- `src/models/model_factory.py` - factory `build_model()` chọn kiến trúc theo `configs/model.yaml -> name` (hiện chỉ hỗ trợ `yolov8`, dễ mở rộng thêm kiến trúc khác sau này)

## 4. Việc cần lưu ý ở bước tiếp theo (05_training.ipynb)

1. Checkpoint hiện tại là **pretrained COCO 80 class** - khi gọi `model.train(data="data/processed/data.yaml", ...)`, ultralytics sẽ tự thay head detection về 1 class (`Human`) theo `data.yaml`, không cần sửa code thủ công.
2. Giữ `imgsz=1280` khi train (không dùng mặc định 640) - đã xác nhận qua smoke test là model chạy được ở resolution này trên GPU 6GB VRAM, cần theo dõi thêm khi train thật với batch_size > 1 (có thể phải giảm batch_size nếu tràn VRAM ở resolution cao).
3. `conf_threshold` khi train xong dùng để inference nên tham khảo `configs/inference.yaml` (đã set 0.15, ưu tiên recall).
