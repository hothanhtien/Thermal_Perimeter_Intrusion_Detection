# Preprocessing

Đã hoàn tất ở `notebooks/03_preprocessing.ipynb` (chạy 2026-07-17). Báo cáo chi tiết:
`reports/preprocessing_report.md`.

## Các bước (theo đúng thứ tự)

1. **Loại ảnh thiếu annotation**: bỏ 64/6.340 ảnh không có file `.txt` tương ứng trong bộ
   annotation chính thức All_In_One (xem `docs/dataset.md`). Không dùng làm negative sample vì
   thiếu file annotation khác với annotation rỗng đã xác nhận 0 box.
2. **Dedupe (MD5)**: gom nhóm ảnh trùng hệt nhau, mỗi nhóm giữ 1 đại diện (tên file nhỏ nhất) để
   tránh cùng một ảnh xuất hiện ở cả train và val/test. Loại thêm 215 ảnh (con số khác "218" ở
   `reports/dataset_intake_report.md` vì ở đây chỉ tính trên 6.276 ảnh đã hợp lệ, không tính trên
   toàn bộ 6.340 ảnh gốc).
3. **Chia train/val/test**: random shuffle (seed 42) theo tỷ lệ 80/10/10 từ `configs/dataset.yaml`.
   Không cần stratify theo class vì single-class, mật độ người thấp và đồng đều.
4. **Tăng cường ảnh nhiệt chuyên biệt** (không phải augmentation ngẫu nhiên, xem
   `src/datasets/preprocessing.py`):
   - Bilateral denoise trước (`d=5, sigma_color=50, sigma_space=50`) - giảm nhiễu cảm biến, giữ
     biên vật thể nhỏ (người ở xa).
   - CLAHE sau (`clip_limit=2.0, tile_grid_size=[8,8]`), áp **trên kênh L của không gian màu LAB**,
     không phải từng kênh R/G/B riêng lẻ - vì ảnh là palette giả màu (false-color, kiểu ironbow/jet)
     áp lên 1 kênh cường độ nhiệt duy nhất (đã verify tương quan R-B âm ~-0,49), CLAHE từng kênh
     riêng sẽ làm lệch màu so với palette gốc.
   - Tham số cấu hình tại `configs/dataset.yaml -> preprocessing`.
5. **Xuất dữ liệu**: ghi ảnh đã tăng cường (không phải bản gốc) vào `data/processed/images/{split}`
   + `data/processed/labels/{split}` theo cấu trúc chuẩn Ultralytics, kèm `data.yaml` (path,
   train/val/test, nc=1, names=[Human]) để `04_model.ipynb`/`05_training.ipynb` load thẳng.

## Kết quả

```
Ảnh gốc: 6.340
  - Loại bỏ (thiếu annotation): 64
  - Loại bỏ (trùng lặp MD5): 215
  - Còn lại: 6.061
Train / Val / Test: 4.849 / 606 / 606
```

`data/processed/manifest.csv` truy vết toàn bộ 6.340 ảnh gốc: giữ lại (kèm split) hay bị loại
(kèm lý do `excluded_missing_annotation` hoặc `dropped_duplicate_of:<file>`).

## Sanity check đã pass

- 0 ảnh trùng MD5 giữa train/val/test (train-val, train-test, val-test đều 0 overlap)
- Số ảnh = số label ở mỗi split, khớp đúng số lượng đã tính trước khi copy

## Lưu ý cho bước sau

- Load dataset qua `data/processed/data.yaml`, không cần parse annotation thủ công.
- Nếu thêm dữ liệu mới (dataset khác), phải chạy lại đúng logic dedupe ở đây để tránh trùng lặp
  giữa các nguồn.
