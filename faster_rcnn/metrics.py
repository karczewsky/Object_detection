import torch
from collections import Counter
from torchvision.ops import box_iou

# Source: https://github.com/PyTorchLightning/pytorch-lightning/pull/4564/files
def object_detection_mean_average_precision(
        preds: torch.Tensor, target: torch.Tensor, iou_threshold: float, ap_calculation: str
) -> torch.Tensor:
    """
    Compute mean average precision for object detection task. Calculates average precision
    using AUC.
    Args:
        preds: an Nx7 batch of predictions with representation
               ``[image_idx, class_pred, class_prob, x_min, y_min, x_max, y_max]``
        target: an Nx6 batch of targets with representation
                ``[image_idx, class_label, x_min, y_min, x_max, y_max]``
        iou_threshold: threshold for IoU score for determining true positive and
                       false positive predictions.
        ap_calculation: one of "step", "VOC2007", "VOC2012", or "COCO"
    Returns:
        mean of the average precision for each class in object detection task.
    """
    if box_iou is None:
        raise ImportError('You want to use `torchvision` which is not installed yet,'
                          ' install it with `pip install torchvision`.')
    classes = torch.cat([preds[:, 1], target[:, 1]]).unique()
    average_precisions = torch.zeros(len(classes))
    for class_idx, c in enumerate(classes):
        c_preds = sorted(preds[preds[:, 1] == c], key=lambda x: x[2], reverse=True)
        c_target = target[target[:, 1] == c]
        targets_per_images = Counter([t[0].item() for t in c_target])
        targets_assigned = {
            image_idx: torch.zeros(count, dtype=torch.bool) for image_idx, count in targets_per_images.items()
        }
        tps = torch.zeros(len(c_preds))
        fps = torch.zeros(len(c_preds))
        if len(c_preds) == 0:
            continue
        for i, p in enumerate(c_preds):
            image_idx = p[0].item()
            ground_truths = c_target[c_target[:, 0] == image_idx]
            ious = box_iou(p[None, 3:], ground_truths[:, 2:])
            best_iou, best_target_idx = ious.squeeze(0).max(0) if len(ground_truths) > 0 else (0, -1)
            if best_iou > iou_threshold and not targets_assigned[image_idx][best_target_idx]:
                targets_assigned[image_idx][best_target_idx] = True
                tps[i] = 1
            else:
                fps[i] = 1
        tps_cum, fps_cum = torch.cumsum(tps, dim=0), torch.cumsum(fps, dim=0)
        precision = tps_cum / (tps_cum + fps_cum)
        recall = tps_cum / len(c_target) if len(c_target) else tps_cum
        precision = torch.cat([reversed(precision), torch.tensor([1.])])
        recall = torch.cat([reversed(recall), torch.tensor([0.])])
        if ap_calculation == "step":
            average_precision = -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])
        elif ap_calculation == "VOC2007":
            average_precision = 0
            recall_thresholds = torch.linspace(0, 1, 11)
            for threshold in recall_thresholds:
                points = precision[:-1][recall[:-1] >= threshold]
                average_precision += torch.max(points) / 11 if len(points) else 0
        elif ap_calculation == "VOC2012":
            average_precision = 0
            for i in range(len(precision)):
                precision[i] = torch.max(precision[:i + 1])
            average_precision = -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])
        elif ap_calculation == "COCO":
            average_precision = 0
            recall_thresholds = torch.linspace(0, 1, 101)
            for threshold in recall_thresholds:
                points = precision[:-1][recall[:-1] >= threshold]
                average_precision += torch.max(points) / 101 if len(points) else 0
        else:
            raise NotImplementedError(f"'{ap_calculation}' is not supported.")
        average_precisions[class_idx] = average_precision
    mean_average_precision = torch.mean(average_precisions)
    return mean_average_precision



# [{'boxes': tensor([[1255.3615, 1008.5653, 1386.4458, 1153.7473],
#         [1133.2731, 1005.9958, 1264.3574, 1152.4625],
#         [1012.4699, 1004.7109, 1144.8393, 1158.8865],
#         [ 284.0161,  623.1264,  529.2571,  756.7451],
#         [ 525.4016,  624.4111,  774.7189,  760.5996],
#         [ 855.6828,  630.8351, 1105.0000,  773.4475],
#         [ 572.9518,  808.1371,  813.2730,  955.8887],
#         [ 359.6186,  811.9914,  579.3775,  950.7495],
#         [ 201.5461,  813.2762,  366.0442,  940.4711],
#         [ 868.5341,  815.8459, 1053.5944,  943.0407],
#         [1049.7389,  814.5610, 1230.9438,  941.7559],
#         [1183.3936,  265.9529, 1374.8796,  423.9829],
#         [1007.3294,  271.0921, 1189.8193,  429.1220],
#         [ 827.4096,  271.0921, 1015.0402,  429.1220],
#         [ 531.8273,  258.2442,  596.0843,  398.2869],
#         [ 476.5663,  258.2442,  538.2530,  395.7173],
#         [ 413.5943,  256.9593,  484.2771,  397.0021],
#         [ 359.6186,  259.5289,  422.5904,  393.1478],
#         [ 548.5342,  403.4261,  615.3614,  535.7602],
#         [ 488.1326,  398.2869,  549.8194,  533.1905],
#         [ 313.3533, 1004.7109,  531.8273, 1151.1777]], device='cuda:0',
#        dtype=torch.float64), 'labels': tensor([ 2,  2,  2,  3,  3,  4,  8,  8,  8, 12, 12, 16, 20, 20, 22, 22, 22, 22,
#         22, 22, 23], device='cuda:0'), 'image_id': tensor([26], device='cuda:0'), 'area': tensor([19031.0859, 19199.5020, 20408.1367, 32768.8125, 33954.1289, 35555.7305,
#         35507.8555, 30493.3242, 20923.3047, 23538.7109, 23048.3223, 30260.5156,
#         28838.8770, 29651.2402,  8998.7402,  8480.2773,  9898.6123,  8414.2188,
#          8843.5234,  8321.7686, 31999.1719], device='cuda:0'), 'iscrowd': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#        device='cuda:0', dtype=torch.uint8)}]
