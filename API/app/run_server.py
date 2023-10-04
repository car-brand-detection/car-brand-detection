import os
import typing as t
import asyncio
import uvicorn
import numpy as np
import aiohttp
import cv2
from fastapi import FastAPI, APIRouter, Header

import torch

# from server_models import detect_cars_on_frame
from server_models.car_model_classifier import predict_car_model

import psutil

process = psutil.Process(os.getpid())


description = """

"""

app = FastAPI(
    title="Synapse API",
    # description=description,
    summary="API детекции и классификации дорожных объектов.",
    contact={
        "name": "Statanly Technologies",
        "url": "https://statanly.com/",
        # "email": "hello@statanly.com",
    },
)
router = APIRouter()
from routers import basic_router
from routers import detector_router, classifier_router
app.include_router(basic_router)
app.include_router(detector_router)
app.include_router(classifier_router)


# app = FastAPI()



async def validate_token(token: str):
    """ Function to validate token. """

    # Set False to check how access declining works
    return True



async def run_web(public=True):
    host = '0.0.0.0' if public else 'localhost'
    config = uvicorn.Config(f"{__name__}:app", host=host, port=9029, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()




if __name__ == "__main__":
    asyncio.run(run_web(public=False))




