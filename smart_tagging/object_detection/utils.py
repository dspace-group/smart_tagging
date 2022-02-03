# Copyright 2022, dSPACE GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you must not use this software except in compliance with the License. This
# software is not fully developed or tested. It is distributed free of charge
# and without any consideration. The software is provided "as is" in the hope
# that it may be useful to other users, but without any warranty of any kind,
# either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

from typing import Tuple

import rtmaps.types
import tensorflow as tf

# color = red + (green * 256) + (blue * 65536)
CLASS_IDS_TO_COLORS = {
    0: 255,       # car: red
    1: 65280,     # pedestrian: green
    2: 16711680,  # truck: blue
    3: 16777215,  # small vehicle: white
    4: 65535,     # utility vehicle: yellow
    5: 16776960,  # bicycle: cyan
    6: 16717055   # tractor: purple
}


def mask_padding(
    boxes: tf.Tensor, scores: tf.Tensor, classes: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Mask zero paddings from model predictions.

    Args:
        boxes (tf.Tensor):
        scores (tf.Tensor):
        classes (tf.Tensor):
    Returns:
        Tuple: tf.Tensor
    """
    mask = tf.greater(tf.reduce_sum(boxes, axis=-1), 0)
    box_count_per_batch = tf.reduce_sum(tf.cast(mask, tf.int16), axis=-1)
    boxes = tf.boolean_mask(boxes, mask).numpy()
    scores = tf.boolean_mask(scores, mask).numpy()
    classes = tf.boolean_mask(classes, mask).numpy()
    return boxes, scores, classes, box_count_per_batch


def get_rtmaps_bbox(
    xmin: float, ymin: float, xmax: float, ymax: float, color: int = 255
) -> rtmaps.types.DrawingObject:
    """
    Creates RTMaps bboxes for the OverlayDrawing block

    Args:
        xmin (float): upper left coordinate
        xmax (float): lower right coordinate
        ymin (float): upper left coordinate
        ymax (float): lower right coordinate
        color (int): color

    Returns:
        Drawing Object: Rectangle
    """
    bbox = rtmaps.types.DrawingObject()
    bbox.kind = 2
    bbox.color = color
    bbox.width = 2
    bbox.data = rtmaps.types.Rectangle()
    bbox.data.x1 = xmin
    bbox.data.y1 = ymin
    bbox.data.x2 = xmax
    bbox.data.y2 = ymax
    return bbox


def get_bbox_annotation(
    x: tf.Tensor,
    y: tf.Tensor,
    label: tf.Tensor,
    score: tf.Tensor,
    color: int = 255,
) -> rtmaps.types.DrawingObject:
    """
    Creates annotations for the bboxes like class names, score.

    Args:
        x (float): x-coordinate
        y (float): y-coordinate
        label (str): text
        score (float): confidence
        color (int): color

    Returns:
        Drawing Object: Text
    """
    text = f'{label} - {str(int(score * 100))}'
    annotation = rtmaps.types.DrawingObject()
    annotation.kind = 5
    annotation.color = color
    annotation.width = 3
    annotation.data = rtmaps.types.Text()
    annotation.data.x = x
    annotation.data.y = y
    annotation.data.cwidth = 15
    annotation.data.cheight = 15
    annotation.data.text = text
    return annotation
