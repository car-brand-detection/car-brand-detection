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


backbone_model_path =  os.path.join(os.getcwd(), 'saved_instances/ArcFace_efficientnet_b2.pth')

label_decoder =  os.path.join(os.getcwd(), "saved_instances/LabelEncoder.pkl")

with open(label_decoder, "rb") as fp:
    label_decoder: LabelEncoder = joblib.load(fp)

labels = torch.tensor(label_decoder.transform(label_decoder.classes_))
num_of_classes = len(label_decoder.classes_)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

backbone_model = torch.load(backbone_model_path, map_location=device).to(device)

classifier = nn.Softmax(dim=-1)

@torch.inference_mode()
def get_softmax_prediction(images, backbone=backbone_model, classifier=classifier, device=device, batch: bool = False):
    backbone.eval()  # Set the model to evaluation mode
    classifier.eval()
    if not batch:
        images = torch.unsqueeze(images, dim=0)

    images = images.to(device)
    logits = backbone(images)
    probabilities: torch.Tensor = classifier(logits)


    confidence, predicted = torch.max(probabilities, 1)
    return predicted.cpu(), confidence.cpu()


def get_car_brand(segmented_car: torch.tensor, method: t.Literal['softmax'] = 'softmax', batch: bool = False) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """ Get predicted label with one of models"""
    if method == 'softmax':
        return get_softmax_prediction(segmented_car, batch=batch)
    else:
        raise NotImplemented


def predict_on_batch(segmented_car: torch.tensor, batch=False) -> str:
    """ Function to predict the car label and decode it to original brand name"""
    if batch:
        labels, confidences = get_car_brand(segmented_car, batch=batch)
        return label_decoder.inverse_transform(labels), confidences
    else:
        label, confidence = get_car_brand(segmented_car,)
        return label_decoder.inverse_transform(label,).item(), confidence.item()


CONF_THRESHOLD = 0.3




def predict_car_labels(cars: np.ndarray) -> list:
    """
    Function to recognize models on batch of cars.
    Each batch item should represent single independent (e.g. detected and extracted) car
    """

    car_brands, confidences = predict_on_batch(segmented_car=cars, batch=True)

    predicted_brands = []

    # for b_id, bbox, car_brand, confidence in zip(b_ids, bboxes, car_brands, confidences):
    #
    #     result = draw_prediction(result, bbox, car_brand, confidence)
    #     predicted_brands.append(car_brand)

    for car_brand, confidence in zip(car_brands, confidences):
        predicted_brands.append(car_brand)

    return predicted_brands




def predict_car_model(images):
    prediction = predict_car_labels(images)
    return prediction