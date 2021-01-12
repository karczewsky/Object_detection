import cv2
import numpy as np
import pytorch_lightning as pl
import torchvision
from torch import Tensor
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path


def infer_image(model: pl.LightningModule, image: Tensor, labels_dict: dict, show_image: bool = False,
                score_threshold: float = 0.5, save_dir: str = None, save_name: str = None):
    """
    Display X-th image in the list with inferred bounding boxes

    Parameters
    ----------
    model : pytorch_lightning.LightningModule model

    image : pytorch.Tensor

    labels_dict :  dict of 'label_id' to 'label'

    show_image : show image

    score_threshold : the minimal score value of boxes visible on the final image, default: 0.5

    save_dir : name of dir in which images should be saved, default: None

    save_name :  name for the new generated image file, default: None
    """
    images = [image]
    sample_image = image.permute(1, 2, 0).cpu().numpy()
    sample_image *= 255
    sample_image = sample_image.astype(np.uint8)

    result = model(images)[0]
    boxes = result['boxes']
    labels = result['labels']
    scores = result['scores']

    nms_boxes_indexes = torchvision.ops.nms(boxes, scores, iou_threshold=0.3).tolist()

    boxes = boxes.cpu().numpy().astype(np.int32)
    labels = labels.cpu().numpy().astype(np.int32)

    boxes_n_labels = list(zip(boxes, labels, scores))

    for box_idx in nms_boxes_indexes:
        (box, label, score) = boxes_n_labels[box_idx]
        if score < score_threshold:
            continue

        text = labels_dict[labels_dict["label_id"] == label]['label'].iloc[0]
        d = {
            "text": text,
            "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
            "fontScale": 0.65,
            "thickness": 2
        }
        (label_width, label_height), baseline = cv2.getTextSize(**d)

        cv2.rectangle(sample_image,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (220, 0, 0),
                      3)

        cv2.rectangle(sample_image,
                      (box[0], box[1]),
                      (box[0] + label_width, box[1] - label_height - 7),
                      (220, 0, 0),
                      -1)

        cv2.putText(sample_image,
                    org=(box[0], box[1] - 5),
                    color=(255, 255, 255),
                    **d)

    if show_image:
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.set_axis_off()
        ax.imshow(sample_image)

    if save_name and save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(sample_image)
        image.save(f"{save_dir}/{save_name}")
