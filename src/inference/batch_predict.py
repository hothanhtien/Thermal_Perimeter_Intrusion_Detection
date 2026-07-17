"""Chạy inference hàng loạt trên 1 thư mục ảnh, lưu annotated image + bảng tổng hợp CSV."""

import csv
import os

from src.inference.predictor import predict_image, save_annotated


def batch_predict(model, input_dir: str, output_dir: str, conf: float, iou: float, imgsz: int, device):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    image_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))

    for fname in image_files:
        image_path = os.path.join(input_dir, fname)
        result = predict_image(model, image_path, conf, iou, imgsz, device)
        save_annotated(result, os.path.join(output_dir, fname))

        confs = result.boxes.conf.tolist() if len(result.boxes) else []
        rows.append({
            "image": fname,
            "num_detections": len(confs),
            "max_confidence": max(confs) if confs else "",
            "mean_confidence": (sum(confs) / len(confs)) if confs else "",
        })

    summary_path = os.path.join(output_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "num_detections", "max_confidence", "mean_confidence"])
        writer.writeheader()
        writer.writerows(rows)

    return rows, summary_path
