# Báo cáo khảo sát dữ liệu (Dataset Intake Report)

Sinh từ `notebooks/01_environment_and_dataset.ipynb` (chạy lại ngày 2026-07-17, kernel `thermal_env`).
Xem `docs/dataset.md` để có tóm tắt ngắn, `configs/dataset.yaml` cho cấu hình chính thức.

## 1. Môi trường

| Hạng mục | Kết quả |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, nhận diện qua `nvidia-smi` (driver 596.36, CUDA 13.2) |
| PyTorch | 2.11.0+cu128, `torch.cuda.is_available() = True`, phép nhân ma trận trên GPU chạy thành công |
| Thư viện | opencv 5.0.0, pillow 12.2.0, numpy 2.2.6, pandas 2.3.3, kagglehub 1.0.2 — đều import được |

=> Môi trường đã sẵn sàng cho bước huấn luyện (GPU hoạt động đúng).

## 2. Dataset

Nguồn: [`animeshmahajan/thermal-image-dataset`](https://www.kaggle.com/datasets/animeshmahajan/thermal-image-dataset), tải qua `kagglehub`, cache tại `~/.cache/kagglehub/datasets/animeshmahajan/thermal-image-dataset/versions/1`.

Tổng quan file: 33.973 file gồm 6.340 ảnh JPEG, 0 video, 16.982 file `.txt` (YOLO, gộp cả bản sao ở root + 2 bộ export), 2 file `coco.json`, 10.646 file `.xml` (VOC).

### 2.1 Ảnh

- Độ phân giải: **100% ảnh 1280x960** (chỉ 1 độ phân giải duy nhất trong toàn bộ dataset)
- Color mode: **100% RGB**
- Đuôi file khớp với định dạng thực tế — không có mismatch
- 0 file 0-byte, 0 ảnh hỏng (không mở được)
- Không có video -> đây là dataset ảnh tĩnh, không áp dụng khái niệm FPS

### 2.2 Metadata

- EXIF: 0/30 ảnh mẫu có dữ liệu EXIF -> **không có** camera model, timestamp, GPS trong file ảnh
- Không tìm thấy file metadata riêng (camera/FOV/khoảng cách/nhiệt độ/thời tiết/timestamp)
- => Dataset không cung cấp điều kiện thu thập (ngày/đêm, khoảng cách lắp đặt, camera). Nếu cần các trường này phải lấy từ tài liệu riêng của tác giả dataset (chưa có trong repo này).

### 2.3 Annotation

Dataset có 3 định dạng song song, tất cả đều single-class **Human**:

| Định dạng | Số file | Số ảnh phủ | Số annotation/bbox |
|---|---|---|---|
| YOLO (gộp cả root + 2 bản export, chỉ để khảo sát) | 16.982 `.txt` | - | 25.823 bbox (đếm trùng bản sao) |
| COCO — All_In_One | 1 `coco.json` | 6.377 ảnh | 9.244 annotation |
| COCO — RGB | 1 `coco.json` | 4.402 ảnh | 7.335 annotation |
| VOC | 10.646 `.xml` | - | 13.614 object |

**Quyết định nguồn annotation chính thức**: dùng bộ **All_In_One** (không dùng bộ "RGB"), vì bao phủ nhiều ảnh hơn (6.276 > 4.370 file annotation YOLO trên tổng 6.340 ảnh). Đã ghi trong `configs/dataset.yaml`.

### 2.4 Tính toàn vẹn dữ liệu

- **Ảnh trùng lặp (MD5)**: 202 nhóm, dư ra **218 file**. Vẫn còn nguyên trong dataset, **chưa được dedupe**.
- **Ảnh thiếu annotation** — có 2 cách tính cho ra 2 con số khác nhau, cần phân biệt rõ:
  - Đối chiếu với txt gốc ở thư mục root: chỉ thiếu **4 ảnh** (`4066`, `6591`, `2932`, `2933`).
  - Đối chiếu với bộ **All_In_One** (nguồn annotation chính thức thực tế đang dùng): thiếu **64 ảnh** (ví dụ: `2381`, `2382`, `2395`, `2396`, `2710`, ...). Lý do: All_In_One là tập con đã lọc của txt root (6.276 vs 6.336 file), ít hơn 60 file so với root.
  - **=> Con số đúng để tham chiếu trong pipeline là 64**, không phải 4. Đã cập nhật `docs/dataset.md` khớp con số này.
  - Sanity check: 5 file mẫu trùng tên giữa root và All_In_One có nội dung giống hệt nhau -> All_In_One đáng tin cậy, chỉ là bản lọc bớt chứ không sai lệch dữ liệu.

## 3. Kết luận & việc cần làm trước khi tiền xử lý

1. Dedupe 202 nhóm ảnh trùng lặp (218 file) trước khi chia train/val/test, tránh rò rỉ dữ liệu (data leakage). **Chưa làm** — sẽ thực hiện ở `03_preprocessing.ipynb`.
2. Quyết định xử lý 64 ảnh thiếu annotation trong bộ All_In_One (loại bỏ hoặc giữ lại làm negative sample). **Chưa làm** — sẽ thực hiện ở `03_preprocessing.ipynb`.
3. Không có metadata điều kiện chụp -> không thể phân tích model theo ngày/đêm, khoảng cách, thời tiết. Đây là hạn chế cố định của dataset, không có việc cần làm thêm.

## 4. Cập nhật sau báo cáo này

- Đã chốt bài toán: **perimeter intrusion detection** (giám sát an ninh/phát hiện xâm nhập ban đêm) — xem `docs/architecture.md`.
- Đã cập nhật `docs/dataset.md` khớp số liệu 64 ảnh thiếu annotation (mục 2.4 ở trên).
- Đã chốt `configs/model.yaml` (yolov8s) và `configs/inference.yaml` (conf_threshold 0.15, ưu tiên recall) theo bài toán trên.
