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

## Scripts

```bash
python scripts/download_dataset.py
python scripts/train.py --config configs/training.yaml
python scripts/evaluate.py --config configs/training.yaml
python scripts/inference.py --config configs/inference.yaml
```
