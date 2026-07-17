"""Tiền xử lý chuyên biệt cho ảnh nhiệt (thermal imaging).

Ảnh trong dataset là ảnh nhiệt đã được colorize bằng palette giả màu (false-color palette)
áp lên một kênh cường độ nhiệt duy nhất - không phải RGB độc lập từng kênh (đã verify:
tương quan R-B âm, đặc trưng của palette kiểu ironbow/jet). Vì vậy khi tăng tương phản
phải làm trên kênh độ sáng (L trong không gian LAB), không phải từng kênh R/G/B riêng lẻ,
nếu không sẽ làm lệch màu so với palette gốc.
"""

import cv2
import numpy as np


def denoise_bilateral(image: np.ndarray, d: int = 5, sigma_color: float = 50,
                       sigma_space: float = 50) -> np.ndarray:
    """Bilateral filter - giảm nhiễu cảm biến nhiệt nhưng vẫn giữ biên vật thể nhỏ (người ở xa)."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                 tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """CLAHE trên kênh L (LAB) - tăng tương phản cục bộ, giữ nguyên quan hệ màu của palette gốc."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid_size))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def preprocess_thermal_image(image: np.ndarray, denoise_params: dict = None,
                              clahe_params: dict = None) -> np.ndarray:
    """Pipeline chuẩn cho ảnh nhiệt: denoise trước (tránh CLAHE khuếch đại nhiễu),
    rồi tăng tương phản cục bộ bằng CLAHE. Không đổi kích thước/dtype ảnh -> bbox
    (toạ độ pixel) vẫn giữ nguyên giá trị sau bước này."""
    image = denoise_bilateral(image, **(denoise_params or {}))
    image = apply_clahe(image, **(clahe_params or {}))
    return image
