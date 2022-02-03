# Copyright 2022, dSPACE GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you must not use this software except in compliance with the License. This
# software is not fully developed or tested. It is distributed free of charge
# and without any consideration. The software is provided "as is" in the hope
# that it may be useful to other users, but without any warranty of any kind,
# either express or implied. See the License for the specific language
# governing permissions and limitations under the License.


from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from smart_tagging.utils import load_exported_model


class ObjectDetection:
    """
    Deploys the exported model and handles preprocessing steps.
    """
    def __init__(
        self,
        model_path: Path,
    ):
        # Model must be an attribute of the class according to
        # https://github.com/tensorflow/tensorflow/issues/46708
        self.model, self.config = load_exported_model(model_path)
        self.serve_fn = self.model.signatures["bboxes"]

    def __call__(
        self, x: np.ndarray, threshold: float
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """

        Args:
            x (tf.Tensor): Input image
            threshold (float): Score threshold of the NMS surpression

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        h = tf.cast(x, tf.float32) / 255.
        h = tf.clip_by_value(h, 0., 1.)
        h = self.serve_fn(image=h, threshold=tf.constant(threshold))
        return h['bboxes'], h['scores'], h['classes'], h['number']
