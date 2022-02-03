# Copyright 2022, dSPACE GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you must not use this software except in compliance with the License. This
# software is not fully developed or tested. It is distributed free of charge
# and without any consideration. The software is provided "as is" in the hope
# that it may be useful to other users, but without any warranty of any kind,
# either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import numpy as np
import rtmaps.types
from rtmaps.base_component import BaseComponent
from smart_tagging import project_root
from smart_tagging.object_detection.model import ObjectDetection
from smart_tagging.object_detection.utils import (
    CLASS_IDS_TO_COLORS,
    get_bbox_annotation,
    get_rtmaps_bbox,
    mask_padding,
)
from smart_tagging.utils import get_ioelt


class ObjectDetectionBlock(BaseComponent):
    """
    RTMaps python bridge that deploys the object detection algorithm
    and provides the detections as well as statistics.
    """
    def __init__(self):
        BaseComponent.__init__(self)

    def Dynamic(self):
        self.add_property("num_images", 1)
        if self.properties["num_images"].data < 1:
            self.properties["num_images"].data = 1
        if self.properties["num_images"].data > 16:
            self.properties["num_images"].data = 16
        for i in range(self.properties["num_images"].data):
            name_in = "image_in_" + str(i)
            name_out = "bbox_" + str(i)
            self.add_input(name_in, rtmaps.types.ANY)
            self.add_output(name_out, rtmaps.types.DRAWING_OBJECT, 200)
        self.add_input("threshold", rtmaps.types.ANY)
        self.add_output("total_num_objects", rtmaps.types.ANY, 16)
        self.add_output("total_num_cars", rtmaps.types.ANY, 16)
        self.add_output("total_num_trucks", rtmaps.types.ANY, 16)
        self.add_output("cur_num_objects", rtmaps.types.ANY, 16)
        self.add_output("cur_num_cars", rtmaps.types.ANY, 16)
        self.add_output("cur_num_trucks", rtmaps.types.ANY, 16)

    def Birth(self):
        self.objects = ObjectDetection(
            project_root / 'object_detection' / 'saved_model')
        self.num_images = self.properties["num_images"].data
        self.labels = {
            v: k.lower()
            for k, v in self.objects.config['model'][
                'dict_class_names_to_ids'].items()}
        self.num_detections = {}
        self.num_detections['total_detections'] = np.array(
            [0] * self.num_images)
        for k in self.objects.config['model']['dict_class_names_to_ids']:
            self.num_detections[k.lower()] = np.array([0] * self.num_images)

    def Core(self):
        # Read inputs.
        threshold = self.inputs["threshold"].ioelt.data
        images_in = []
        boxes_out = []
        resolutions = []
        for i in range(self.num_images):
            name_in = "image_in_" + str(i)
            input_data = self.inputs[name_in].ioelt.data
            input_ts = self.inputs[name_in].ioelt.ts
            images_in.append(input_data.image_data.copy())
            resolutions.append(input_data.image_data.shape[:2])
            boxes_out.append(get_ioelt(input_ts, []))
        images = np.stack(images_in, axis=0)

        # Perform object detection and format output.
        boxes, scores, classes, num_detections = self.objects(
            images, threshold=threshold / 100)
        boxes, scores, classes, box_count_per_batch = mask_padding(
            boxes, scores, classes)

        # Prepare bounding boxes.
        old_statistic = self.num_detections.copy()
        image_id = 0
        id = 0
        Y, X = resolutions[image_id]
        for i in range(boxes.shape[0]):
            id += 1

            # Check to which image in the batch this detection belongs.
            if id > box_count_per_batch[image_id]:
                image_id += 1
                id = 0
                Y, X = resolutions[image_id]

            # Create box and label for detection.
            xmin, ymin, xmax, ymax = np.split(boxes[i], 4, axis=-1)
            xmin, xmax = xmin[0] * X, xmax[0] * X
            ymin, ymax = ymin[0] * Y, ymax[0] * Y
            c = int(classes[i])
            bbox = get_rtmaps_bbox(
                xmin, ymin, xmax, ymax, color=CLASS_IDS_TO_COLORS[c]
            )
            annotation = get_bbox_annotation(
                xmin, ymin - 20, self.labels[c], scores[i],
                color=CLASS_IDS_TO_COLORS[c]
            )
            boxes_out[image_id].data.append(bbox)
            boxes_out[image_id].data.append(annotation)
            self.num_detections[self.labels[c]][image_id] += 1

        num_detections = num_detections.numpy()
        self.num_detections['total_detections'] += num_detections

        # Create output elements.
        num_objects = get_ioelt(
            input_ts, self.num_detections['total_detections'])
        num_cars = get_ioelt(input_ts, self.num_detections['car'])
        num_trucks = get_ioelt(input_ts, self.num_detections['truck'])
        cur_objects = get_ioelt(input_ts, num_detections)
        cur_cars = get_ioelt(
            input_ts, self.num_detections['car'] - old_statistic['car'])
        cur_trucks = get_ioelt(
            input_ts, self.num_detections['truck'] - old_statistic['truck'])

        # Write output elements.
        for i in range(self.num_images):
            # If there are no detections for this frame, add an empty box.
            if len(boxes_out[i].data) == 0:
                bbox = get_rtmaps_bbox(0., 0., 0., 0.)
                boxes_out[i].data.append(bbox)
            name_out = "bbox_" + str(i)
            self.outputs[name_out].write(boxes_out[i])
        self.outputs["total_num_objects"].write(num_objects)
        self.outputs["total_num_cars"].write(num_cars)
        self.outputs["total_num_trucks"].write(num_trucks)
        self.outputs["cur_num_objects"].write(cur_objects)
        self.outputs["cur_num_cars"].write(cur_cars)
        self.outputs["cur_num_trucks"].write(cur_trucks)

    def Death(self):
        pass
