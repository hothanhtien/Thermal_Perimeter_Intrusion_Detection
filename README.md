# Thermal AI Project

## Bài toán

**Giám sát an ninh / phát hiện xâm nhập ban đêm bằng camera nhiệt** (perimeter intrusion detection).
Camera nhiệt cố định (static), không cần ánh sáng, phát hiện người (single-class **Human**) xuất hiện
trong khu vực giám sát để cảnh báo xâm nhập.

- Input: ảnh từ camera nhiệt cố định, 1280x960
- Output: bounding box + confidence cho từng người phát hiện trong khung hình
- Ưu tiên: **recall cao hơn precision** - bỏ lọt người xâm nhập (false negative) nguy hiểm hơn báo động
  giả (false positive) trong bài toán an ninh
- Ràng buộc: người trong ảnh thường nhỏ (xa camera) - xem `docs/dataset.md` và notebook `02_dataset_analysis.ipynb`

Xây dựng pipeline AI: khảo sát dữ liệu, tiền xử lý, huấn luyện, đánh giá và inference trên
[`animeshmahajan/thermal-image-dataset`](https://www.kaggle.com/datasets/animeshmahajan/thermal-image-dataset).

## Môi trường

- Conda env: `thermal_env` (Python 3.10)
- GPU: yêu cầu CUDA (đã verify với NVIDIA RTX 3060, PyTorch cu128)

```bash
conda env create -f environment.yml
conda activate thermal_env
```

hoặc dùng pip:

```bash
pip install -r requirements.txt
```

## Cấu trúc thư mục

```
configs/      cấu hình YAML (dataset, training, model, inference)
data/         dữ liệu (raw, external, interim, processed, annotations, cache, sample)
notebooks/    notebook khảo sát / thí nghiệm, đánh số thứ tự theo pipeline
src/          source code chính (datasets, models, losses, metrics, training, evaluation, inference, visualization, utils)
reports/      báo cáo, figures, tables sinh ra từ quá trình phân tích
outputs/      checkpoint, prediction, log, tensorboard
tests/        unit test cho src/
scripts/      entrypoint CLI (download, train, evaluate, inference, export)
docs/         tài liệu chi tiết (kiến trúc, dataset, preprocessing, experiments)
```

## Quy trình

1. `notebooks/01_environment_and_dataset.ipynb` - kiểm tra môi trường/GPU/CUDA + tải dataset + khảo sát format/metadata/annotation/tính toàn vẹn
2. `notebooks/02_dataset_analysis.ipynb` - EDA chi tiết
3. `notebooks/03_preprocessing.ipynb` - tiền xử lý + chia train/val/test
4. `notebooks/04_model.ipynb` - định nghĩa/chọn model
5. `notebooks/05_training.ipynb` - training + validation
6. `notebooks/06_evaluation.ipynb` - đánh giá + error analysis
7. `notebooks/07_inference.ipynb` - inference + export model

## Tổng quan quá trình thực hiện

Đã hoàn thành cả 7 bước của pipeline (chạy thật, verify từng bước, chi tiết xem trong `reports/`):

**01 - Environment & Dataset**: xác nhận GPU RTX 3060 + PyTorch CUDA hoạt động đúng. Tải dataset qua
`kagglehub` (6.340 ảnh JPEG 1280x960, không video, không EXIF/metadata điều kiện chụp). Dataset có 2 bản
annotation song song ("All_In_One" và "RGB") - **quyết định dùng All_In_One** vì bao phủ nhiều ảnh hơn.
Phát hiện 202 nhóm ảnh trùng lặp (MD5) và 64 ảnh thiếu annotation cần xử lý ở bước sau.

**02 - Dataset Analysis**: EDA chi tiết trên 9.244 bbox (single-class `Human`) - mật độ người thấp
(1-3 người/ảnh), bbox nhỏ và có 2 cụm khoảng cách (gần/xa camera), dải sáng ảnh hẹp (~70-140/255).

**03 - Preprocessing**: loại 64 ảnh thiếu annotation (không dùng làm negative sample vì không chắc chắn),
dedupe 215 ảnh trùng lặp, áp dụng **bilateral denoise + CLAHE trên kênh L (LAB)** - kỹ thuật chuyên biệt cho
ảnh nhiệt (đã verify đây là palette giả màu áp lên 1 kênh cường độ, không phải RGB độc lập). Chia
train/val/test = 4.849/606/606, xuất theo cấu trúc chuẩn YOLOv8 tại `data/processed/`.

**04 - Model**: khởi tạo YOLOv8s pretrained COCO qua factory pattern (`src/models/`), verify chạy được
trên GPU, đối chiếu cấu hình model khớp `data.yaml`.

**05 - Training**: phát hiện **GPU 6GB VRAM bị OOM ở `imgsz=1280`** (kể cả dùng autobatch của ultralytics)
- đã đổi xuống **`imgsz=640`** sau khi trao đổi với người dùng (đánh đổi: giảm chi tiết người ở xa để đổi
lấy khả năng chạy được trên phần cứng hiện có). Smoke test 2 epoch chạy thành công (batch=4 tự động,
~150s/epoch), ước tính full training 100 epoch ~4,2 giờ. **Full training thật chưa chạy** (chờ quyết định
người dùng - xem `reports/training_report.md` để biết cách bật).

**06 - Evaluation**: pipeline đánh giá trên tập test + error analysis riêng theo kích thước bbox (do lo
ngại ảnh hưởng của việc đổi imgsz) - đã verify chạy đúng với checkpoint tạm (2-epoch smoke test), kết quả
ban đầu cho thấy recall không giảm nghiêm trọng ở bbox nhỏ.

**07 - Inference & Export**: inference đơn ảnh + hàng loạt, export ONNX thành công (`best.onnx`, 44,7 MB)
- sẵn sàng triển khai lên edge device không cần Python/PyTorch đầy đủ.

**Trạng thái hiện tại**: toàn bộ pipeline đã chạy thông từ đầu đến cuối bằng checkpoint tạm (2 epoch).
Việc còn lại: chạy full training thật (100 epoch), rồi chạy lại notebook 06 và 07 để có kết quả cuối cùng
chính xác (không cần sửa code, tự động dùng checkpoint mới đã ghi đè `outputs/checkpoints/best.pt`).

## Scripts

```bash
python scripts/download_dataset.py
python scripts/train.py --config configs/training.yaml
python scripts/evaluate.py --config configs/training.yaml
python scripts/inference.py --config configs/inference.yaml
```
