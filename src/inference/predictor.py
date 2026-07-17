"""Predictor: load checkpoint, chạy inference trên 1 ảnh, lưu ảnh đã vẽ box."""

import cv2
from ultralytics import YOLO


def load_predictor(checkpoint_path: str) -> YOLO:
    return YOLO(checkpoint_path)


def predict_image(model: YOLO, image_path: str, conf: float, iou: float, imgsz: int, device):
    """Chạy predict trên 1 ảnh, trả về object Results (ultralytics) đầu tiên."""
    results = model.predict(source=image_path, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)
    return results[0]


def save_annotated(result, output_path: str) -> None:
    """Lưu ảnh đã vẽ sẵn bounding box + confidence (do ultralytics tự vẽ) ra output_path."""
    annotated = result.plot()  # BGR numpy array
    cv2.imwrite(output_path, annotated)
