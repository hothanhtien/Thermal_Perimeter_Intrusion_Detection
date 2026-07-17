"""Thống kê đánh giá: IoU matching giữa prediction và ground truth, recall theo kích thước bbox."""


def box_iou(box1, box2):
    """IoU giữa 2 box dạng (x1, y1, x2, y2), toạ độ pixel."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_predictions_to_gt(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Greedy IoU matching - mỗi GT box khớp với tối đa 1 pred box (và ngược lại).

    Trả về list bool cùng độ dài gt_boxes: True nếu GT đó được phát hiện (matched), False nếu bị bỏ lọt.
    """
    matched_pred = set()
    matched_gt = [False] * len(gt_boxes)
    for gi, gt in enumerate(gt_boxes):
        best_iou = 0.0
        best_pi = -1
        for pi, pred in enumerate(pred_boxes):
            if pi in matched_pred:
                continue
            iou = box_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_pi = pi
        if best_iou >= iou_threshold and best_pi >= 0:
            matched_gt[gi] = True
            matched_pred.add(best_pi)
    return matched_gt


def recall_by_size_bucket(records, bucket_edges):
    """records: list dict {"sqrt_area": float, "matched": bool}.
    bucket_edges: danh sách biên tăng dần, vd [0, 40, 60, 100, 300].
    Trả về dict {(lo, hi): {"n": so luong GT, "recall": ty le matched}}.
    """
    buckets = {}
    for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
        subset = [r for r in records if lo <= r["sqrt_area"] < hi]
        n = len(subset)
        recall = sum(r["matched"] for r in subset) / n if n else float("nan")
        buckets[(lo, hi)] = {"n": n, "recall": recall}
    return buckets
