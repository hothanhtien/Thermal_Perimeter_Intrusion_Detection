# Kiến trúc dự án

## Bài toán (chốt ngày 2026-07-17)

**Perimeter intrusion detection** - phát hiện người xâm nhập ban đêm bằng camera nhiệt cố định.

| Hạng mục | Quyết định | Lý do |
|---|---|---|
| Task | Object detection, single-class `Human` | Annotation gốc chỉ có 1 class, phù hợp dùng YOLOv8 detect thay vì segmentation |
| Triển khai | Camera nhiệt tĩnh (không di chuyển) | Heatmap vị trí bbox ở `02_dataset_analysis.ipynb` cho thấy phân bố vị trí ổn định, đặc trưng camera cố định |
| Metric ưu tiên | Recall > Precision (dùng recall@fixed-FP hoặc tune threshold theo recall) | An ninh: bỏ lọt người xâm nhập (FN) nguy hiểm hơn báo động giả (FP) |
| Object size | Nhỏ - phần lớn người ở xa (sqrt-area trung bình ~63px / 1280x960) | Ảnh hưởng chọn anchor/imgsz - xem `configs/training.yaml` (image_size giữ nguyên 1280x960, không resize xuống 640 để tránh mất chi tiết người nhỏ) |
| Model | YOLOv8 (variant nhỏ/vừa: n hoặc s) | GPU RTX 3060 Laptop 6GB VRAM - không đủ để train yolov8l/x hiệu quả |

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
