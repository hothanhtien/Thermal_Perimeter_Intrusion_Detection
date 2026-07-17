# Kiến trúc dự án

## Bài toán (chốt ngày 2026-07-17)

**Perimeter intrusion detection** - phát hiện người xâm nhập ban đêm bằng camera nhiệt cố định.

| Hạng mục | Quyết định | Lý do |
|---|---|---|
| Task | Object detection, single-class `Human` | Annotation gốc chỉ có 1 class, phù hợp dùng YOLOv8 detect thay vì segmentation |
| Triển khai | Camera nhiệt tĩnh (không di chuyển) | **Đính chính (2026-07-17)**: heatmap vị trí bbox ở `02_dataset_analysis.ipynb` KHÔNG chứng minh 1 camera cố định duy nhất - dataset không có metadata xác nhận điều này, và vị trí bbox tập trung ở nửa dưới khung hình chỉ phản ánh phối cảnh camera nhìn xuống (đúng với hầu hết camera an ninh, không đặc trưng riêng 1 vị trí lắp đặt). Xem mục "Giới hạn dataset" bên dưới |
| Metric ưu tiên | Recall > Precision (dùng recall@fixed-FP hoặc tune threshold theo recall) | An ninh: bỏ lọt người xâm nhập (FN) nguy hiểm hơn báo động giả (FP) |
| Object size | Nhỏ - phần lớn người ở xa (sqrt-area trung bình ~63px / 1280x960) | Ảnh hưởng chọn anchor/imgsz - ban đầu định giữ nguyên 1280x960, nhưng **đã đổi xuống imgsz=640** (2026-07-17) vì GPU 6GB VRAM OOM ở 1280 ngay cả với autobatch (verify thực tế ở `05_training.ipynb`) - xem `configs/training.yaml`. Đánh đổi: giảm chi tiết người ở xa, cần theo dõi ảnh hưởng recall ở `06_evaluation.ipynb` |
| Model | YOLOv8 (variant nhỏ/vừa: n hoặc s) | GPU RTX 3060 Laptop 6GB VRAM - không đủ để train yolov8l/x hiệu quả, và ở imgsz=1280 còn không đủ cho cả yolov8s |

## Giới hạn dataset & phạm vi dự án (2026-07-17)

Dataset/model trong dự án này chỉ giải quyết **core detector** (nhận diện người trên 1 ảnh nhiệt đơn lẻ).
Một hệ thống giám sát an ninh hoàn chỉnh cần thêm tầng ứng dụng nằm **ngoài phạm vi dự án**, vì 2 giới hạn
sau của dataset:

1. **Không có frame liên tục (video)** - dataset 100% ảnh tĩnh độc lập, không FPS, không timestamp nối
   các ảnh (xem `01_environment_and_dataset.ipynb`). Hệ quả: không thể tracking quỹ đạo di chuyển, không
   thể multi-frame confirmation (kỹ thuật phổ biến để giảm false alarm bằng cách chỉ báo động khi phát
   hiện liên tục qua N frame), không thể phân biệt "đi vào" vs "đi ngang qua" khu vực. Model chỉ làm được
   single-frame detection - mỗi ảnh xử lý độc lập, không có trạng thái giữa các lần phát hiện.

2. **Không thể học ROI (Region of Interest) cố định từ dataset** - vì không có metadata xác nhận 1 camera
   vật lý cố định duy nhất, không thể định nghĩa 1 vùng cảnh báo xâm nhập trong toạ độ pixel rồi áp dụng
   chung cho mọi camera thật. ROI bắt buộc phải cấu hình thủ công riêng cho từng camera lúc triển khai.

### Khuyến nghị khi triển khai inference thực tế

Vì dataset training không có video, tầng ứng dụng thực tế (ngoài phạm vi dự án) nên bổ sung một bước
**tiền lọc candidate ROI** trước khi chạy YOLOv8 detector, thay vì quét full-frame mỗi lần:

1. **Nếu camera thật có video liên tục (frame + metadata/timestamp liên tục)** - dùng thuật toán
   **MOG2** (`cv2.createBackgroundSubtractorMOG2`, background subtraction kiểu Mixture of Gaussians) để mô
   hình hoá nền tĩnh một cách thích nghi (tự cập nhật theo thời gian, chịu được thay đổi nhiệt độ nền
   chậm/từ từ) và tách phần tiền cảnh (foreground mask) mỗi frame - đây là thuật toán chuẩn, có sẵn trong
   OpenCV, phù hợp hơn cách tự tính "trung bình động + ngưỡng" thủ công vì đã xử lý tốt nhiễu và thích nghi
   theo thời gian.
2. Từ foreground mask của MOG2, tìm contour/bounding box của các vùng tiền cảnh đủ lớn (lọc theo diện tích
   tối thiểu để loại nhiễu) - đó chính là các **candidate ROI** (khả năng cao chứa vật thể chuyển động/người,
   vì nhiệt độ người khác nền tĩnh).
3. Chỉ chạy model YOLOv8 (đã train ở dự án này) trên các candidate ROI đó (crop + resize), không chạy
   full-frame - đây là kiến trúc 2 giai đoạn phổ biến trong giám sát video thời gian thực (candidate
   proposal rẻ bằng MOG2 -> detector đắt chỉ chạy trên vùng nghi ngờ).

**Nếu không có video liên tục** (chỉ có ảnh tĩnh rời rạc, như dataset huấn luyện của dự án này) - MOG2
không áp dụng được vì nó cần chuỗi frame theo thời gian để học nền. Khi đó bắt buộc phải chạy full-frame
detector trên từng ảnh, không có bước tiền lọc.

Lợi ích khi có MOG2: giảm compute (tăng FPS trên edge device), tận dụng được tín hiệu chuyển động mà model
single-frame không có, và bù đắp một phần cho việc thiếu multi-frame confirmation + ROI cố định đã nêu ở
trên.

## Pipeline tổng quan

```
data intake -> preprocessing -> split -> training -> evaluation -> inference -> export
```

## Module chính (src/)

- `datasets/` - load, tiền xử lý, augmentation, transform
- `models/` - kiến trúc model (CNN, ResNet, UNet, DeepLab, YOLOv8) + factory
- `losses/` - hàm loss (Dice, Focal, MSE) + factory
- `metrics/` - accuracy, precision, recall, IoU, Dice, mAP, confusion matrix
- `training/` - trainer, scheduler, optimizer, callbacks
- `evaluation/` - evaluate, visualize, report, statistics
- `inference/` - predictor, batch predict, export
- `visualization/` - plots, histogram, overlay, heatmap
- `utils/` - logger, config, seed, file/image IO, metadata, helpers

## Config-driven

Mỗi bước (dataset/model/training/inference) đọc tham số từ `configs/*.yaml`, không hardcode trong code.
