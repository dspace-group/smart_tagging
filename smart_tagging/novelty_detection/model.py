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

import numpy as np
import tensorflow as tf

from smart_tagging.utils import load_exported_model


class PairwiseFilter:
    """
    Compute pairwise similarity of two sequencial images.
    """

    def __init__(
        self,
        model_path: Path
    ):
        self.model, self.config = load_exported_model(model_path)
        self.serve_fn = self.model.signatures["features"]
        self.buffer = None

    def cosine_similarity(
        self, x: tf.Tensor, y: tf.Tensor
    ) -> tf.Tensor:
        """

        Args:
            x (tf.Tensor): Feature vector
            y (tf.Tensor): Feature vector

        Returns:
            tf.Tensor: Cosine similarity
        """
        return tf.reduce_sum(
            tf.nn.l2_normalize(x, axis=-1) * tf.nn.l2_normalize(y, axis=-1),
            axis=-1
        )

    def __call__(self, x: np.ndarray) -> tf.Tensor:
        """
        Computes similarity of the prior an current image.

        Args:
            x (tf.Tensor): Input image

        Returns:
            tf.Tensor: Similarity
        """
        h = tf.cast(tf.expand_dims(x, axis=0), tf.float32) / 255.

        if self.buffer is None:
            features = self.serve_fn(image=h)
            self.buffer = features['feature']
            return tf.zeros((1,))
        else:
            features = self.serve_fn(image=h)['feature']
            sim = self.cosine_similarity(features, self.buffer)
            self.buffer = features
            return sim
