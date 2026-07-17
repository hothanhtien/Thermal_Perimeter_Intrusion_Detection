# Dataset: thermal-image-dataset

Nguồn: https://www.kaggle.com/datasets/animeshmahajan/thermal-image-dataset

## Tóm tắt khảo sát (xem notebooks/01_environment_and_dataset.ipynb)

- 6.340 ảnh JPEG, độ phân giải cố định 1280x960, RGB (ảnh nhiệt đã colorize, không phải radiometric 16-bit)
- Không có video, không có EXIF/camera model/FOV/điều kiện môi trường/timestamp
- Annotation: YOLO (.txt), COCO (coco.json), VOC (.xml) - đều single-class **Human**
- 202 nhóm ảnh trùng lặp (MD5), 64 ảnh thiếu annotation YOLO trong bộ All_In_One (nguồn chính thức - xem `configs/dataset.yaml`)

## Lưu ý khi sử dụng

- Dedupe trước khi chia train/val/test để tránh data leakage
- Hai bộ annotation export ("All_In_One" và "RGB") có thể chứa tập con trùng lặp - cần đối chiếu file_name trước khi gộp
