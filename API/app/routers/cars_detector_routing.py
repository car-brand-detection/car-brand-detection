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

from aiohttp import MultipartReader
import torch

# from server_models import detect_cars_on_frame
# from server_models.car_model_classifier import predict_car_model
from server_models.car_detector import detect_cars_on_frame
# from server_models.utils import extract_car, get_batch_of_images



import psutil
import torch.cuda

process = psutil.Process(os.getpid())


description = """

"""


router = APIRouter()

# app = FastAPI()



async def validate_token(token: str):
    """ Function to validate token. """

    # Set False to check how access declining works
    return True


async def cars_streamer(data: t.List):
    for item in data:
        yield item



@router.post("/api/detect-cars")
async def detect_cars(
        file: UploadFile = File(...)
) -> JSONResponse:
    try:
        start_time = time.time()
        image_bytes = await file.read()

        # Convert the bytes into a NumPy array
        image = np.frombuffer(image_bytes, np.uint8)  # Type is uint8

        # Decode the NumPy array as an image (adjust the format as needed)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # cars: t.List[np.ndarray] = detect_cars_on_frame(image)
        dots: t.List[t.Dict] = detect_cars_on_frame(image)
        return dict(
            success=True,
            data=dots,
            duration=time.time() - start_time,
        )

        # ---------- ANOTHER METHOD
        headers = {
            "success": "True",
            "X-bytes_size": []
        }
        result = []
        for car in cars:
            car = cv2.flip(car, 1)
            _, image = cv2.imencode(".jpg", car)

            image = image.tobytes()
            headers["X-bytes_size"].append(len(image))
            result.append(image)
        headers["X-bytes_size"] = str(json.dumps(headers["X-bytes_size"]))

        result = cars_streamer(result)
        return StreamingResponse(content=result,
                                 headers=headers,
                                 # headers={"X-bytes_size": "YES"},
                                 media_type="image/jpeg",
                                 )
        return {
            "success": True,
            "data": result,
            "duration": time.time() - start_time,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "success": False,
            "message": str(e)
        })





