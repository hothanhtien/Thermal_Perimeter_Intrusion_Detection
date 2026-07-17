.PHONY: env download train evaluate inference test lint clean

env:
	conda env create -f environment.yml

download:
	python scripts/download_dataset.py

train:
	python scripts/train.py --config configs/training.yaml

evaluate:
	python scripts/evaluate.py --config configs/training.yaml

inference:
	python scripts/inference.py --config configs/inference.yaml

test:
	pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
