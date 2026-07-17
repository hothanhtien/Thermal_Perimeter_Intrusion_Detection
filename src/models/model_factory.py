"""Factory chọn kiến trúc model theo `configs/model.yaml` (trường `name`)."""

from src.models.yolov8 import build_yolov8

_BUILDERS = {
    "yolov8": build_yolov8,
}


def build_model(cfg: dict, **kwargs):
    name = cfg["name"]
    if name not in _BUILDERS:
        raise ValueError(f"Model '{name}' chưa được hỗ trợ. Các lựa chọn: {list(_BUILDERS)}")
    return _BUILDERS[name](cfg, **kwargs)
