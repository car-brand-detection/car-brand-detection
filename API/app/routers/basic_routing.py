import json
import os

from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Header, Form
from fastapi.responses import StreamingResponse, Response, JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, Response
from pydantic import BaseModel, Json
import json

from aiohttp import MultipartReader
import torch


import psutil
import torch.cuda

process = psutil.Process(os.getpid())

class Root(BaseModel):
    is_alive: bool = True

class HealthReport(BaseModel):
    self: str = "|Alive: True|-|Memory: 0.0 MiB|-|CPU usage: 0.0|-|Threads: 0|-|Device: Cuda|"

router = APIRouter()
device = (
    "Cuda"
    if torch.cuda.is_available()
    else "MPS"
    if torch.backends.mps.is_available()
    else "CPU"
)

@router.get("/")
def ping_root() -> Root:
    """ Проверка доступности сервера. """
    return {"is_alive": True}


@router.get("/health", tags=["health"])
def view_health():
    """
    Проверка работы сервера и доступных ресурсов.
    """
    data = {
        "Alive": True,
        "Memory": f"{process.memory_info().rss / (1024 ** 2):.3f} MiB",
        "CPU usage": process.cpu_percent(),
        "Threads": len(process.threads()),
        "Device": device,
    }
    answer = "|" + "|-|".join(map(
        lambda x: f"{x[0]}: {str(x[1])}",
        list(data.items()))
    ) + "|"
    return answer
