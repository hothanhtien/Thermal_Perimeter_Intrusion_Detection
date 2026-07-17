"""CLI: đánh giá model trên tập test theo configs/inference.yaml.

Tương đương notebooks/06_evaluation.ipynb mục "metric tổng quan" (model.val() chuẩn ultralytics).
"""

import argparse
import os

import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Đánh giá model trên tập test")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--data", default="data/processed/data.yaml")
    args = parser.parse_args()

    with open(os.path.join(PROJECT_ROOT, args.config), encoding="utf-8") as f:
        infer_cfg = yaml.safe_load(f)

    checkpoint = os.path.join(PROJECT_ROOT, infer_cfg["checkpoint"])
    data_yaml = os.path.join(PROJECT_ROOT, args.data)
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(checkpoint)
    metrics = model.val(
        data=data_yaml,
        split="test",
        imgsz=infer_cfg["image_size"][0],
        conf=infer_cfg["conf_threshold"],
        iou=infer_cfg["iou_threshold"],
        device=device,
    )

    print("\nPrecision:", metrics.box.p[0])
    print("Recall:", metrics.box.r[0])
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()
