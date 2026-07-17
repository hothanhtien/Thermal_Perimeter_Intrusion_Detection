"""CLI: fine-tune YOLOv8 theo configs/training.yaml + configs/model.yaml.

Tương đương phần training của notebooks/05_training.ipynb, dùng lại src/models/model_factory.py.
"""

import argparse
import os
import shutil
import sys

import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.models.model_factory import build_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 trên thermal dataset")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--data", default="data/processed/data.yaml")
    args = parser.parse_args()

    with open(os.path.join(PROJECT_ROOT, args.config), encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    with open(os.path.join(PROJECT_ROOT, args.model_config), encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    weights_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
    logs_dir = os.path.join(PROJECT_ROOT, "outputs", "logs")
    data_yaml = os.path.join(PROJECT_ROOT, args.data)
    device = 0 if torch.cuda.is_available() else "cpu"

    model = build_model(model_cfg, weights_dir=weights_dir)

    # batch=-1 (autobatch): GPU VRAM han che tung gay OOM o batch co dinh, ke ca sau khi
    # giam thu cong - xem reports/training_report.md muc 2-3 de biet ly do.
    model.train(
        data=data_yaml,
        epochs=train_cfg["training"]["epochs"],
        imgsz=train_cfg["training"]["image_size"][0],
        batch=-1,
        workers=train_cfg["training"]["num_workers"],
        amp=train_cfg["training"]["amp"],
        optimizer=train_cfg["optimizer"]["name"].upper(),
        lr0=train_cfg["optimizer"]["lr"],
        weight_decay=train_cfg["optimizer"]["weight_decay"],
        cos_lr=(train_cfg["scheduler"]["name"] == "cosine"),
        warmup_epochs=train_cfg["scheduler"]["warmup_epochs"],
        patience=train_cfg["early_stopping"]["patience"],
        seed=train_cfg["seed"],
        device=device,
        project=logs_dir,
        name="yolov8s_thermal",
        exist_ok=True,
    )

    best_src = os.path.join(logs_dir, "yolov8s_thermal", "weights", "best.pt")
    checkpoint_dst = os.path.join(weights_dir, "best.pt")
    if os.path.isfile(best_src):
        shutil.copy2(best_src, checkpoint_dst)
        print("Đã copy best checkpoint sang:", checkpoint_dst)


if __name__ == "__main__":
    main()
