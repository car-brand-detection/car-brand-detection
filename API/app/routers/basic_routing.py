import json
import os

from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Header, Form
from fastapi.responses import StreamingResponse, Response, JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, Response
import json

from aiohttp import MultipartReader
import torch


import psutil
import torch.cuda

process = psutil.Process(os.getpid())


router = APIRouter()

@router.get("/")
def ping_root():
    """ Проверка доступности сервера. """
    return {"Is alive": True}


@router.get("/health", tags=["health"])
def view_health() -> Response:
    """
    Проверка работы сервера и доступных ресурсов.
    """
    data = {
        "Alive": True,
        "Memory": f"{process.memory_info().rss / (1024 ** 2):.3f} MiB",
        "CPU usage": process.cpu_percent(),
        "Threads": len(process.threads()),
        "Is CUDA available": torch.cuda.is_available(),
    }
    answer = "|" + "|-|".join(map(
        lambda x: f"{x[0]}: {str(x[1])}",
        list(data.items()))
    ) + "|"
    return answer
