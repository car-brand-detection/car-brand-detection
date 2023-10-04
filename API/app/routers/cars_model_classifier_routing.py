import json
import os
import typing as t
import time
import asyncio
import uvicorn
import numpy as np
import aiohttp
import cv2
from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Header, Form, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Json

class ClassifierResponse(BaseModel):
    success: bool = True
    data: t.List[str,] = ['model1','model2']
    duration: float = 0.0


# from server_models import detect_cars_on_frame
from server_models.car_model_classifier import predict_car_model
from utils import extract_car, get_batch_of_images

router = APIRouter()

async def cars_streamer(data: t.List):
    for item in data:
        yield item


file_description = "Кадр с автомобилями (в формате uint8 байт-кода)"
json_data_description = """Координаты автомобилей на кадре следующего формата:
- на вход отправляется список (python list);
- каждый элемент в списке представляет один автомобиль и является словарём (python dict);
- словарь имеет два ключа: `box` и `points`;
- `box` - список из четырёх чисел (рамка объекта);
- `point` - массив точек контура объекта, размерности 2xN, N - произвольное, в среднем 50 < N < 300;
"""


@router.post("/api/get-car-model", )
async def recognize_cars_on_frame(
        json_data: str = Form(..., description=json_data_description),
        file: UploadFile = File(..., description=file_description)
) -> ClassifierResponse:
    """
    Определение модели авто на изображении.
    Input:
        1) оригинальный кадр с камеры (на котором автомобиль обнаружен);
        2) боксы и маски всех обнаруженных автомобилей в формате JSON (данные от сегментатора).
    Модель итеративно "приближает" каждый автомобиль на исходном кадре, и затем полученный батч
    изображений отправляется в классификатор.
    Output: предсказанные модели всех обнаруженных машин (в той же последовательности, в который были отправлены боксы).
    """
    dots = json.loads(json_data)['dots']
    try:
        start_time = time.time()
        image_bytes = await file.read()

        # Convert the bytes into a NumPy array
        image = np.frombuffer(image_bytes, np.uint8)

        # Decode the NumPy array as an image (adjust the format as needed)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        cars = []
        errors = []
        for i, car in enumerate(dots):
            box, points = car['box'], car['points']
            if box and points: # Not empty lists
                box, points = np.array(box).astype(np.int32), np.array(points).astype(np.int32)
                car: np.ndarray = extract_car(image, bbox=box, points=points)
                cars.append(car)
            else:
                errors.append(i)

        cars = get_batch_of_images(cars)
        labels: list = predict_car_model(cars)  # Inference on batch

        [labels.insert(i, "Unknown") for i in errors]
        return JSONResponse(content={
            "success": True,
            "data": labels,
            "duration": time.time() - start_time,
        }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "success": False,
            "message": str(e)
        })






#
# @app.post("/api/get-cars-on-photo")
# async def detect_and_recognize_cars(
#         # token: str = Header(None),
#         response: Request,
#         file: UploadFile = File(...)
# ):
#     if not await validate_token(
#             response.headers['authorization']
#     ):
#         raise HTTPException(
#             status_code=401,
#             detail={
#                 "success": False,
#                 "message": "Authorization failed! Token is invalid."
#             }
#         )
#
#     try:
#         start_time = time.time()
#         image_bytes = await file.read()
#
#         # Convert the bytes into a NumPy array
#         image = np.frombuffer(image_bytes, np.uint8)
#
#         # Decode the NumPy array as an image (adjust the format as needed)
#         image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#
#         # image = cv2.flip(image, 1)
#
#         cars = detect_cars_on_frame(image)
#         cars = torch.from_numpy(np.stack(cars, axis=0))
#         # cars = torch.cat(cars, dim=0)
#
#         labels = predict_car_model(cars)
#         return {
#             "success": True,
#             "data": labels,
#             "message": f"Request processed successfully!",
#             "duration": time.time() - start_time,
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail={
#             "success": False,
#             "message": str(e)
#         })


