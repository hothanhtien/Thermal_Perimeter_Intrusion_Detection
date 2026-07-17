"""Export model sang định dạng triển khai (ONNX...) - dùng cho inference ngoài môi trường ultralytics/PyTorch."""

from ultralytics import YOLO


def export_model(model: YOLO, format: str = "onnx", imgsz: int = 640):
    """Trả về đường dẫn file đã export."""
    return model.export(format=format, imgsz=imgsz)
