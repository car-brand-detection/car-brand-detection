""" That is bogus car detector. I use it until we get real one. """

# -*- coding: utf-8 -*-

# Major builtin libraries
import os
import typing as t
from copy import deepcopy

# Classic packages for data manipulation and visualization
import numpy as np

# Basic PyTorch
import torch

# Torch Computer Vision tools for images processing
from ultralytics import YOLO


# from ..utils import draw_rectangle
yolo_model_path = os.path.join(os.getcwd(), "saved_instances/yolov8x-seg.pt")


def get_max_area(indexed_box1, indexed_box2):
    """
    Comparing function that will be passed to 'reduce'.
    We calculate area of two boxes and return biggest
    :param indexed_box1:
    :param indexed_box2:
    :return: one of boxes - tuple like `index, bbox`
    """
    b1_i, (b1_x, b1_y, b1_x2, b1_y2) = indexed_box1
    b2_i, (b2_x, b2_y, b2_x2, b2_y2) = indexed_box2
    b1_area = (b1_x2 - b1_x) * (b1_y2 - b1_y)
    b2_area = (b2_x2 - b2_x) * (b2_y2 - b2_y)

    # print(f"B1 AREA: {b1_area}")
    result = indexed_box1 if b1_area > b2_area else indexed_box2
    return result


class YOLOSegmentation:
    def __init__(self, model_path, device, accepted_ids: list):
        self.device = device
        self.model = YOLO(model_path,)
        self.required_classes = accepted_ids

    def detect(self, img):
        if img is None:
            return [], [], [], []
        height, width, channels = img.shape

        results = self.model.predict(
            source=img, save=False, save_txt=False, device=self.device, verbose=False,
            conf=0.25  # Default confidence is 0.25. We need more so that we will skip various trimmed machines that we don't expect to get
        )
        # print(results)
        if results is None:
            return [], [], [], []

        result = results[0]
        segmentation_contours_idx = []

        if result.masks is None:
            return [], [], [], []

        for seg in result.masks.xyn:
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentation_contours_idx.append(segment)
        bboxes = np.array(result.boxes.xyxy.to('cpu'), dtype="int")
        class_ids = np.array(result.boxes.cls.to('cpu'), dtype="int")
        scores = np.array(result.boxes.conf.to('cpu'), dtype="float").round(2)

        return bboxes, class_ids, segmentation_contours_idx, scores

    def remove_small_boxes(self, bboxes, use_toral_area=True):
        """

        :param bboxes:
        :param use_toral_area: whether we use two conditions (for both A and B) or one - for rectangle AxB
        :return:
        """
        id_s = []
        bboxes = list(bboxes)
        for b_id, bbox in enumerate(bboxes):
            x, y, x2, y2 = bbox
            a, b = x2-x, y2-y

            if (use_toral_area and a * b >= 160**2):
                # (not use_toral_area and a >= 160 and b >= 160)
                id_s.append(b_id)
            else:
                bboxes.pop(b_id)
        return id_s, bboxes

    def select_biggest_car(self, bboxes,):
        """

        :param bboxes:
        :param use_toral_area: whether we use two conditions (for both A and B) or one - for rectangle AxB
        :return:
        """

        bboxes = list(bboxes)
        x, y, x2, y2 = bboxes[0]
        a, b = x2-x, y2-y
        max_area = a * b
        max_id = 0

        for b_id, bbox in enumerate(bboxes):
            x, y, x2, y2 = bbox
            area = (x2-x) * (y2-y)
            if area >= max_area:
                max_area = area
                max_id = b_id

        bbox = bboxes[max_id]

        return [max_id, ], [bbox, ]


    def select_cars(self, bboxes, classes, filter: t.Literal['all', 'big', 'biggest']):
        self.max_area = 0
        proper_boxes = []
        boxes_id = []
        for b_id, (object_class, bbox) in enumerate(zip(classes, bboxes)):
            # Calculate number of objects that may be cars
            if object_class in self.required_classes:
                proper_boxes.append(bbox)
                boxes_id.append(b_id)

        if filter == 'big':
            boxes_id, bboxes = self.remove_small_boxes(bboxes)
        elif filter == 'biggest':
            boxes_id, bboxes = self.select_biggest_car(bboxes)

        return boxes_id, bboxes

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

yolo_segmentator = YOLOSegmentation(yolo_model_path, device,
    accepted_ids=[2, 5, 7]  # Car, Bus, Truck
)



CONF_THRESHOLD = 0.3



def detect_cars_on_frame(frame: np.ndarray, Segmentator: YOLOSegmentation = yolo_segmentator, detect: t.Literal['all', 'big', 'biggest']='big') -> dict:
    """ Function to detect cars on image
    :returns batch of cars

    """
    if frame is None:
        return dict(image=None, label="_")
    result = deepcopy(frame)

    bboxes, classes, segmentations, scores = Segmentator.detect(frame)
    b_ids, bboxes = Segmentator.select_cars(bboxes=bboxes, classes=classes, filter=detect)


    # cars = []
    dots = []
    for b_id, bbox in zip(b_ids, bboxes):  # Dtype of bbox and segmentations is np.int32
        # cars.append(draw_rectangle(image=result, bbox=bbox))
        # cars.append(extract_car(image=result, bbox=bbox, points=np.array(segmentations[b_id])))
        dots.append({
            "box": bbox.tolist(),
            "points": np.array(segmentations[b_id]).tolist()
        })

    return dots
