"""CLI: chạy inference hàng loạt theo configs/inference.yaml.

Tương đương notebooks/07_inference.ipynb mục "inference hàng loạt", dùng lại src/inference/.
"""

import argparse
import os
import sys

import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.inference.batch_predict import batch_predict  # noqa: E402
from src.inference.predictor import load_predictor  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Inference hàng loạt trên thư mục ảnh")
    parser.add_argument("--config", default="configs/inference.yaml")
    args = parser.parse_args()

    with open(os.path.join(PROJECT_ROOT, args.config), encoding="utf-8") as f:
        infer_cfg = yaml.safe_load(f)

    checkpoint = os.path.join(PROJECT_ROOT, infer_cfg["checkpoint"])
    input_dir = os.path.join(PROJECT_ROOT, infer_cfg["input_dir"])
    output_dir = os.path.join(PROJECT_ROOT, infer_cfg["output_dir"])
    device = 0 if torch.cuda.is_available() else "cpu"

    model = load_predictor(checkpoint)
    rows, summary_path = batch_predict(
        model, input_dir=input_dir, output_dir=output_dir,
        conf=infer_cfg["conf_threshold"], iou=infer_cfg["iou_threshold"],
        imgsz=infer_cfg["image_size"][0], device=device,
    )

    print(f"Đã xử lý {len(rows)} ảnh, kết quả tại {output_dir}")
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
