import json
import os
import typing as t
import time
import asyncio
import uvicorn
import numpy as np
import aiohttp
import cv2
from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Header, Form
from fastapi.responses import StreamingResponse, Response, JSONResponse
from pydantic import BaseModel, Json

from aiohttp import MultipartReader
import torch


from server_models.car_detection import detect_cars_on_frame, segment_cars_on_frame, segment_cars_on_batch



router = APIRouter()

# async def cars_streamer(data: t.List):
#     for item in data:
#         yield item


class DetectorResponse(BaseModel):
    success: bool = True
    data: t.List[t.Dict] = [{
        "box": [0, 1, 1, 2],
    }]
    duration: float = 0.0


class SegmentatorResponse(BaseModel):
    success: bool = True
    data: t.List[t.Dict] = [{
        "box": [0, 1, 1, 2],
        "points": [[0,0], [1,1], [2,2], [3,3],]
    }]
    duration: float = 0.0


file_4_detector_description = "Кадр с автомобилями (uint8 байт-код), на котором детектор будет искать автомобили. "
file_4_exctractor_description = "Кадр с автомобилями (uint8 байт-код), на котором сегментатор будет извлекать автомобили. "

json_data_description = """Координаты автомобилей на кадре следующего формата:
- на вход отправляется список (python list);
- каждый элемент в списке представляет один автомобиль и является словарём (python dict);
- словарь имеет один ключ - `box`;
- `box` - список из четырёх чисел (рамка объекта);
- `box` используется для приближения кадра для каждого автомобиля.
"""

@router.post("/api/detect-cars")
async def detect_cars(
        file: UploadFile = File(..., description=file_4_detector_description)
) -> DetectorResponse:
    """
    Детекция автомобилей на изображении.
    Input: кадр с камеры;
    Output: боксы всех обнаруженных машин.
    """
    try:
        start_time = time.time()
        image_bytes = await file.read()

        # Convert the bytes into a NumPy array
        image = np.frombuffer(image_bytes, np.uint8)  # Type is uint8

        # Decode the NumPy array as an image (adjust the format as needed)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        dots: t.List[t.Dict] = detect_cars_on_frame(image)

        return dict(
            success=True,
            data=dots,
            duration=time.time() - start_time,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "success": False,
            "message": str(e)
        })


@router.post("/api/extract-cars")
async def extract_cars(
        file: UploadFile = File(..., description=file_4_exctractor_description),
        json_data: str = Form(..., description=json_data_description),
) -> SegmentatorResponse:
    """
    Детекция автомобилей на изображении.
    Input:
        1) кадр с камеры;
        2) боксы всех обнаруженных автомобилей в формате JSON (данные от детектора).
    Output: боксы и маски всех обнаруженных машин.
    """

    dots = json.loads(json_data)['dots']
    try:
        start_time = time.time()
        image_bytes = await file.read()

        # Convert the bytes into a NumPy array
        image = np.frombuffer(image_bytes, np.uint8)

        # Decode the NumPy array as an image (adjust the format as needed)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        cars: t.List[t.Dict] = []
        shifts: t.List[t.Tuple[int, int]] = []


        for i, car in enumerate(dots):
            box = car['box']
            box = np.array(box).astype(np.int32)
            (x, y, x2, y2) = box
            shifts.append((x, y))
            cars.append(image[y:y2, x:x2])

        cars: t.Dict[str, t.List] = segment_cars_on_batch(cars, box_shifts=shifts)

        return JSONResponse(content={
            "success": True,
            "data": cars,
            "duration": time.time() - start_time,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "success": False,
            "message": str(e)
        })



