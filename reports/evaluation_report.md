# Báo cáo đánh giá model (Evaluation Report)

Sinh từ `notebooks/06_evaluation.ipynb` (chạy ngày 2026-07-17, kernel `thermal_env`).

**QUAN TRỌNG**: checkpoint dùng để đánh giá là `outputs/checkpoints/best.pt` - đây là checkpoint **tạm
2-epoch smoke test** (xem `reports/training_report.md`), **chưa phải kết quả full training thật**. Toàn
bộ số liệu dưới đây chỉ mang tính minh hoạ pipeline evaluation chạy đúng, không phải đánh giá chất lượng
model cuối cùng. Cần chạy lại notebook này sau khi full training 100-epoch xong.

## 1. Metric tổng quan (tập test, 606 ảnh)

| Metric | Giá trị |
|---|---|
| Precision | 0,967 |
| Recall | 0,973 |
| mAP50 | 0,968 |
| mAP50-95 | 0,597 |

Nguồn: `model.val()` chuẩn của ultralytics, `conf=0.15`, `iou=0.45` (theo `configs/inference.yaml`).

## 2. Error analysis theo kích thước bbox

Tự chạy `predict` + đối chiếu IoU (threshold 0,5) với ground truth gốc (toạ độ pixel theo resolution
1280x960, độc lập với `imgsz` lúc train) để kiểm tra ảnh hưởng của việc đổi `imgsz` 1280→640
(xem `reports/training_report.md` mục 2) lên khả năng phát hiện người nhỏ/xa.

- Tổng số GT box đối chiếu: 904
- Recall tổng thể (tính tay, đối chiếu với `model.val()`): 0,979 - khớp hợp lý với 0,973 ở trên

| Nhóm sqrt(area) bbox | Số lượng | Recall |
|---|---|---|
| 0 - 40 px | 226 | 0,969 |
| 40 - 51,3 px | 226 | 0,987 |
| 51,3 - 74,6 px | 226 | 0,965 |
| 74,6 - 227,7 px | 226 | 0,996 |

**Nhận xét**: recall không sụt giảm nghiêm trọng ở nhóm bbox nhỏ nhất (0,969) so với nhóm lớn nhất (0,996)
- chênh lệch ~2,7 điểm phần trăm, không phải mức đáng báo động. Đây là tín hiệu ban đầu tích cực rằng việc
đổi `imgsz` 1280→640 (bắt buộc do giới hạn 6GB VRAM) **có thể không ảnh hưởng quá nặng** đến phát hiện người
ở xa - nhưng **cần xác nhận lại với checkpoint full training thật**, vì model mới học 2 epoch chưa đủ để
kết luận chắc chắn (có thể recall ở nhóm nhỏ sẽ phân hoá rõ hơn khi model hội tụ đầy đủ).

## 3. Ảnh mẫu (ground truth vs prediction)

Đã kiểm tra trực quan 6 ảnh mẫu ngẫu nhiên từ tập test (xanh lá = ground truth, đỏ = prediction) - bbox dự
đoán bám sát vị trí người thật, không lệch đáng kể.

## 4. Code liên quan

- `src/evaluation/statistics.py` - `box_iou()`, `match_predictions_to_gt()` (greedy IoU matching),
  `recall_by_size_bucket()`
- `src/evaluation/visualize.py` - `draw_boxes()` để overlay GT/prediction lên ảnh

## 5. Việc cần làm sau khi full training thật xong

1. Chạy lại `06_evaluation.ipynb` (không cần sửa code - tự động dùng checkpoint mới ở
   `outputs/checkpoints/best.pt` sau khi ghi đè).
2. So sánh lại recall theo bucket kích thước bbox - xác nhận nhận xét ở mục 2 có còn đúng không.
3. Nếu recall nhóm bbox nhỏ giảm đáng kể so với nhóm lớn, cân nhắc: tăng lại `imgsz` nếu có GPU VRAM lớn
   hơn (cloud), hoặc dùng kỹ thuật tiling/multi-scale inference cho các vùng nghi có người ở xa.
4. Cập nhật lại bảng metric ở mục 1 và biểu đồ recall-theo-size với số liệu thật.
