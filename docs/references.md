# References

- Dataset: [animeshmahajan/thermal-image-dataset](https://www.kaggle.com/datasets/animeshmahajan/thermal-image-dataset)
- PyTorch: https://pytorch.org
- Ultralytics YOLOv8 (model + training/eval/export API dùng xuyên suốt `src/models`, `src/inference`, `scripts/`): https://docs.ultralytics.com
- OpenCV CLAHE (dùng trong `src/datasets/preprocessing.py`): https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
- OpenCV MOG2 background subtraction (đề xuất cho tầng ứng dụng thực tế, xem `docs/architecture.md`): https://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html
