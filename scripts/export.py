"""CLI: export model sang định dạng triển khai (ONNX mặc định) theo configs/inference.yaml."""

import argparse
import os
import sys

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.inference.export import export_model  # noqa: E402
from src.inference.predictor import load_predictor  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Export model sang định dạng triển khai")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--format", default="onnx")
    args = parser.parse_args()

    with open(os.path.join(PROJECT_ROOT, args.config), encoding="utf-8") as f:
        infer_cfg = yaml.safe_load(f)

    checkpoint = os.path.join(PROJECT_ROOT, infer_cfg["checkpoint"])
    model = load_predictor(checkpoint)
    out_path = export_model(model, format=args.format, imgsz=infer_cfg["image_size"][0])
    print("Đã export:", out_path)


if __name__ == "__main__":
    main()
