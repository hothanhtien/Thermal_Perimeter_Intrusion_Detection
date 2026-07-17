"""Vẽ ground truth / prediction lên ảnh để kiểm tra trực quan kết quả evaluation."""

from matplotlib.patches import Rectangle


def draw_boxes(ax, boxes, color, label=None):
    """Vẽ danh sách box (x1, y1, x2, y2) lên axes matplotlib."""
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        ax.add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor=color, linewidth=1.5,
            label=label if i == 0 else None,
        ))
