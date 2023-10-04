
# -*- coding: utf-8 -*-

# Major builtin libraries
import os
import typing as t
from copy import deepcopy

# Classic packages for data manipulation and visualization
import numpy as np

# Basic PyTorch
import torch
import torch.nn as nn

# Utils
import joblib  # Pipelining, pickling (dump/load), parallel processing
from tqdm import tqdm  # Progress bar for training process
# Classic ML tools
from sklearn.preprocessing import LabelEncoder

# Torch Computer Vision tools for images processing
import cv2
from ultralytics import YOLO

# Albumentations is an OS library for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from albumentations import Resize, Normalize

CONF_THRESHOLD = 0.3


data_normalization = A.Compose([
    A.Resize(256, 256),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0
    ),
    ToTensorV2()], p=1.
)



def draw_prediction(original_image, bbox, label, confidence) -> np.ndarray:
    """ Function to draw a rectangle and car label on the source frame."""


    (x, y, x2, y2) = bbox
    if confidence < 0.3:
        color = (200, 0, 0)
    elif confidence < 0.8:
        color = (230, 230, 0)

    else:
        color = (100, 255, 0)

    cv2.rectangle(original_image, (x, y), (x2, y2), color, 2)
    text = "{}: {:.1f}%".format(label, 100*confidence)

    text_size, _ = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3)
    text_w, text_h = text_size
    cv2.rectangle(original_image, (x, y), (x+text_w, y+text_h), color, -1)

    if confidence < CONF_THRESHOLD:
        text = "Undefined (<{}%)".format(100*CONF_THRESHOLD)
    cv2.putText(img=original_image,
                text=text,
                org=(x, y+text_h),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2
                )
    return original_image





def draw_rectangle(image, bbox, label=None) -> np.ndarray:
    """
    Function to draw a rectangle around the car.
    If label passed, puts it to the image too.
    """
    image = deepcopy(image)


    (x, y, x2, y2) = bbox
    color = (100, 255, 0)

    cv2.rectangle(image, (x, y), (x2, y2), color, 2)


    if label:
        text = "{}".format(label,)

        text_size, _ = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)
        text_w, text_h = text_size
        cv2.rectangle(image, (x, y), (x+text_w, y+text_h), color, -1)
        # cv2.rectangle(image, (x, y+text_h), (x+text_w, y),  color, -1)


        cv2.putText(img=image,
                    text=text,
                    org=(x, y+text_h),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2
                    )
    else:
        text_h = 0

    image = image[y - text_h:y2, x:x2]
    return image


def extract_car(image, bbox, points) -> np.ndarray:
    image = deepcopy(image)
    if bbox is None:
        return None

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

    (x, y, x2, y2) = bbox

    car = cv2.bitwise_and(image, image, mask=mask)
    car = car[y:y2, x:x2]  # dtype is uint8

    return car


def get_batch_of_images(cars: t.Sequence[np.ndarray]) -> torch.Tensor:
    data = []
    for car in cars:
        data.append(data_normalization(image=car)['image'])
    return torch.stack(data, dim=0)
