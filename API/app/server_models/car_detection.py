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

yolo_seg_model = os.path.join(os.getcwd(), "saved_instances/yolov8n-seg.pt")
yolo_base_model = os.path.join(os.getcwd(), "saved_instances/yolov8n.pt")


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

    result = indexed_box1 if b1_area > b2_area else indexed_box2
    return result


class YOLOModel:
    def __init__(self, model_path, device, accepted_ids: list, yolo_kind=t.Literal['detector', 'segmentator'], imsize=128):
        self.yolo_kind = yolo_kind
        self.device = device
        self.size=imsize
        self.model = YOLO(model_path,)
        self.required_classes = accepted_ids

    def detect(self, img):
        if img is None:
            return [], [], [], []
        height, width, channels = img.shape

        results = self.model.predict(
            source=img, save=False, save_txt=False, device=self.device, verbose=False, imgsz=self.size,
            conf=0.25  # Default confidence is 0.25. We need more so that we will skip various trimmed machines that we don't expect to get
        )
        if results is None:
            return [], [], [], []

        result = results[0]
        segmentation_contours_idx = []

        if result.masks is not None:
            # So it's not a basic YOLO model, but a segmentator
            for seg in result.masks.xyn:
                seg[:, 0] *= width
                seg[:, 1] *= height
                segment = np.array(seg, dtype=np.int32)
                segmentation_contours_idx.append(segment)
        bboxes = np.array(result.boxes.xyxy.to('cpu'), dtype="int")
        class_ids = np.array(result.boxes.cls.to('cpu'), dtype="int")
        scores = np.array(result.boxes.conf.to('cpu'), dtype="float").round(2)

        return bboxes, class_ids, segmentation_contours_idx, scores


    def segment_on_batch(self, batch) -> t.Dict[str, t.List]:
        """ Function to detect cars on batch of images. """

        results = self.model.predict(
            source=batch, save=False, save_txt=False, device=self.device, verbose=False, imgsz=self.size,
            conf=0.25  # Default confidence is 0.25. We need more so that we will skip various trimmed machines that we don't expect to get
        )
        if results is None:
            return [], [], [], []

        answer_data: t.Dict[str, t.List] = {
            "bboxes": [],
            "points": []
        }

        for result in results:
            i = 0
            height, width = result.orig_shape

            if result.masks is not None:
                # So it's not a basic YOLO model, but a segmentator
                seg = result.masks.xyn[0]
                i += 1
                seg[:, 0] *= width
                seg[:, 1] *= height
                segment = np.array(seg, dtype=np.int32)
                answer_data['points'].append(segment)

                bboxes = np.squeeze(np.array(result.boxes.xyxy.to('cpu'), dtype="int"))
                answer_data['bboxes'].append(bboxes)
            else:
                # Empty
                answer_data['points'].append(None)
                answer_data['bboxes'].append(None)




                #
                # for seg in result.masks.xyn:
                #     print("I: ", i)
                #     i += 1
                #     seg[:, 0] *= width
                #     seg[:, 1] *= height
                #     segment = np.array(seg, dtype=np.int32)
                #     answer_data['points'].append(segment)
                #
                #     bboxes = np.array(result.boxes.xyxy.to('cpu'), dtype="int")
                #     answer_data['bboxes'].append(bboxes)

        return answer_data

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

yolo_base_model = YOLOModel(yolo_base_model, device,
                                    accepted_ids=[2, 5, 7],  # Car, Bus, Truck
                                    yolo_kind='detector',
                                    imsize=(640, 480),
                                    )

yolo_seg_model = YOLOModel(yolo_seg_model, device,
                        accepted_ids=[2, 5, 7],  # Car, Bus, Truck
                        yolo_kind='segmentator',
                        imsize=128
                        )



CONF_THRESHOLD = 0.3



def detect_cars_on_frame(frame: np.ndarray, Detector: YOLOModel = yolo_base_model, detect: t.Literal['all', 'big', 'biggest']='big') -> dict:
    """ Function to detect cars on image

    """
    if frame is None:
        return dict(image=None, label="_")

    bboxes, classes, segmentations, scores = Detector.detect(frame)
    b_ids, bboxes = Detector.select_cars(bboxes=bboxes, classes=classes, filter=detect)

    dots = []
    for b_id, bbox in zip(b_ids, bboxes):  # Dtype of bbox and segmentations is np.int32
        dots.append({
            "box": bbox.tolist(),
        })

    return dots


def segment_cars_on_frame(frame: np.ndarray,
                          segmentator: YOLOModel = yolo_seg_model,
                          shift_by=(0,0),
    ) -> dict:
    """
     Function to segment cars on image.

    :param frame: cropped pieces of original
    :param segmentator: Yolo-seg model, which is able to extract objects from the image.
    :param shift_by: If we pass cropped image we have to shift our points, so they will match the original picture
    :return:
    """
    if frame is None:
        return dict(image=None, label="_")

    bboxes, classes, segmentations, scores = segmentator.detect(frame)
    b_ids, bboxes = segmentator.select_cars(bboxes=bboxes, classes=classes, filter="biggest")

    dots = []
    for b_id, bbox in zip(b_ids, bboxes):  # dtype of bbox and segmentations is np.int32
        x, y = shift_by
        points = np.array(segmentations[b_id]) + np.array(shift_by)

        bbox[0] += x  # Shift
        bbox[1] += y
        bbox[2] += x
        bbox[3] += y

        dots.append({
            "box": bbox.tolist(),
            "points": points.tolist()
        })

    return dots



def segment_cars_on_batch(images: t.List[np.ndarray],
                          segmentator: YOLOModel = yolo_seg_model,
                          box_shifts = [(0, 0)],
                          ) -> dict:
    """
     Function to segment cars on image.

    :param frame: cropped pieces of original
    :param segmentator: Yolo-seg model, which is able to extract objects from the image.
    :param box_shifts: If we pass cropped image we have to shift our points, so they will match the original picture
    :return:
    """

    data = segmentator.segment_on_batch(images)
    bboxes, segments = data["bboxes"], data["points"]

    dots = []
    for bbox, points, shift in zip(bboxes, segments, box_shifts):  # dtype of bbox and segmentations is np.int32
        # x, y = shift_by
        x, y = shift
        if bbox is not None and points is not None:
            points = points + np.array(shift)

            bbox[0] += x  # Shift
            bbox[1] += y
            bbox[2] += x
            bbox[3] += y

            dots.append({
                "box": bbox.tolist(),
                "points": points.tolist()
            })
        else:
            dots.append({
                "box": [],
                "points": []
            })

    return dots
