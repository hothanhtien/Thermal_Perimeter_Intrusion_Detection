"""Xây dựng model YOLOv8 (ultralytics) theo cấu hình dự án."""

import os
import shutil

from ultralytics import YOLO

DEFAULT_WEIGHTS_DIR = "outputs/checkpoints"


def build_yolov8(cfg: dict, weights_dir: str = DEFAULT_WEIGHTS_DIR) -> YOLO:
    """Load checkpoint YOLOv8 theo variant/pretrained trong `configs/model.yaml`.

    cfg: dict đọc từ configs/model.yaml, ví dụ {"variant": "s", "pretrained": True, ...}
    weights_dir: nơi cache checkpoint pretrained - mặc định "outputs/checkpoints" (nên truyền
    đường dẫn tuyệt đối khi gọi từ notebook, vì cwd của kernel là notebooks/). Tránh để
    ultralytics tự tải .pt vào bất kỳ thư mục hiện tại nào mỗi lần chạy.
    """
    variant = cfg.get("variant", "s")

    if not cfg.get("pretrained", True):
        return YOLO(f"yolov8{variant}.yaml")

    filename = f"yolov8{variant}.pt"
    os.makedirs(weights_dir, exist_ok=True)
    cached_path = os.path.join(weights_dir, filename)

    if not os.path.isfile(cached_path):
        YOLO(filename)  # ultralytics tự động tải `filename` vào cwd nếu chưa có
        shutil.move(filename, cached_path)

    return YOLO(cached_path)
