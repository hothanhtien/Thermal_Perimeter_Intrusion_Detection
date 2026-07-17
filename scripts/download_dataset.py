"""CLI: tải dataset thermal-image-dataset qua kagglehub (xem configs/dataset.yaml)."""

import argparse
import os

import kagglehub
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Tải dataset thermal-image-dataset")
    parser.add_argument("--config", default="configs/dataset.yaml")
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config)
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    slug = cfg["dataset"]["kaggle_slug"]
    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "kagglehub", "datasets", *slug.split("/"), "versions"
    )
    cached_versions = os.path.isdir(cache_dir) and os.listdir(cache_dir)

    if cached_versions:
        path = os.path.join(cache_dir, sorted(cached_versions)[-1])
        print("Dataset đã có sẵn trong cache:", path)
    else:
        path = kagglehub.dataset_download(slug)
        print("Đã tải dataset về:", path)


if __name__ == "__main__":
    main()
