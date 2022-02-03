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
import rtmaps.types
from rtmaps.base_component import BaseComponent
from smart_tagging import project_root
from smart_tagging.novelty_detection.model import PairwiseFilter
from smart_tagging.utils import get_ioelt


class NoveltyFilter(BaseComponent):
    """
    Inherits from the RTMaps python bridge template.
    Can be used as a single block in the RTMaps diagram.
    """
    def __init__(self):
        BaseComponent.__init__(self)
        self.buffer_img = None

    def Dynamic(self):
        self.add_input("image_in", rtmaps.types.ANY)
        self.add_output("similarity", rtmaps.types.AUTO, 1)

    def Birth(self):
        self.novelty_module = PairwiseFilter(
            project_root / 'novelty_detection' / 'saved_model'
        )

    def Core(self):

        image = self.inputs["image_in"].ioelt.data.image_data.copy()
        input_ts = self.inputs["image_in"].ioelt.ts

        sim = self.novelty_module(image).numpy()
        sim = 1. - np.arccos(sim) / np.pi * 2
        sim = sim.astype(np.float32)

        similarity = get_ioelt(input_ts, sim)

        self.outputs["similarity"].write(similarity)

    def Death(self):
        pass
