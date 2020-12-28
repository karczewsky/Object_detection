import torch
from collections import Counter
from torchvision.ops import box_iou


def object_detection_mean_average_precision(
    pred: torch.Tensor, target: torch.Tensor, iou_threshold: float, num_classes: int
) -> torch.Tensor:
    """
    Compute mean average precision for object detection task. Calculates average precision
    using AUC.
    Args:
        pred: Tensor containing predictions, where each row is of the following format:
              [image_idx, class_pred, class_prob, x_min, y_min, x_max, y_max]
        target: Tensor containing ground truths, where each row is of the following format:
                [image_idx, class_label, x_min, y_min, x_max, y_max]
        iou_threshold: threshold for IoU score for determining true positive and
                       false positive predictions.
        num_classes: number of total classes
    Returns:
        mean of the average precision for each class in object detection task.
    """
    average_precisions = torch.zeros(num_classes)
    for c in range(num_classes):
        c_pred = [p for p in pred if p[1] == c]
        c_target = [t for t in target if t[1] == c]
        if len(c_target) == 0:
            continue
        # Sort c_pred in descending order w.r.t. confidence score
        c_pred = sorted(c_pred, key=lambda x: x[2], reverse=True)
        num_targets_per_image = Counter(t[0].item() for t in c_target)
        targets_tracker = {
            image_idx: torch.zeros(count)
            for image_idx, count in num_targets_per_image.items()
        }
        tps, fps = torch.zeros(len(c_pred)), torch.zeros(len(c_pred))
        for i, p in enumerate(c_pred):
            # Get the targets that correspond to the same image as the prediction
            ground_truths = [t for t in c_target if t[0] == p[0]]
            best_iou = 0
            best_target_idx = 0
            for j, t in enumerate(ground_truths):
                curr_iou = box_iou(p[None, 3:], t[None, 2:])
                if curr_iou > best_iou:
                    best_iou = curr_iou
                    best_target_idx = j
            image_idx = int(p[0].item())
            if (
                best_iou > iou_threshold
                and targets_tracker[image_idx][best_target_idx] == 0
            ):
                targets_tracker[image_idx][best_target_idx] = 1
                tps[i] = 1
            else:
                fps[i] = 1
        tps_cum, fps_cum = torch.cumsum(tps, dim=0), torch.cumsum(fps, dim=0)
        precision = tps_cum / (tps_cum + fps_cum)
        recall = tps_cum / len(c_target)
        precision = torch.cat([reversed(precision), torch.tensor([1])])
        recall = torch.cat([reversed(recall), torch.tensor([0])])
        average_precision = -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])
        average_precisions[c] = average_precision
    return torch.mean(average_precisions)
