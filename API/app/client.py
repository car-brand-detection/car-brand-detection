import os
import sys
import json
import typing as t
import asyncio
import numpy as np
import aiohttp
import cv2
from aiohttp import StreamReader, MultipartWriter
from aiohttp.multipart import CIMultiDictProxy



import logging
logger = logging.getLogger()
# logger = logging.Logger(logging.basicConfig())
# from fastapi.logger import

from utils import extract_car, draw_rectangle, draw_prediction
from run_server import run_web

async def show_progress(current=0, total=30, size=50, item='file'):
    string_pointer = int(current / total * size)
    label = f"|{item} {current}|"
    label = f"{label:->{string_pointer + len(label)}}"
    label = f"{label:><{size - string_pointer + len(label)}}"
    label += f"|of {total}|"
    print(f"""\033[32m\n{label}\033[39m""")

async def show_frame(frame, windom_name='Window') -> None:
    """ Function to show window with video stream. """
    cv2.namedWindow(windom_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windom_name, 600, 400)
    cv2.imshow(windom_name, frame)


async def send_request_to_classifier(image: np.ndarray,
                                     json_data: dict,
                       url = "http://localhost:9029/api/get-car-model"
                       ):
    async with aiohttp.ClientSession() as session:
        # Prepare the POST request with the image data

        data = aiohttp.FormData()
        _, image = cv2.imencode(".jpg", image)
        image = image.tobytes()
        data.add_field("file", image, filename="image.jpg", content_type="image/jpeg")

        data.add_field('json_data', json.dumps(json_data), content_type='application/json')

        async with session.post(url,
                                # headers=headers,
                                data=data
                                ) as response:
            response_json = await response.json()
            if response.status == 200:
                return response_json
            else:
                error = response_json['detail']
                print(f"Failed to send POST request. Status code: {response.status}. "
                      f"Error message: {error}"
                      )





async def send_batch_to_classifier(images: t.List[np.ndarray],
                                     json_data: dict,
                                     url = "http://localhost:9029/api/get-car-model"
                                     ):
    async with aiohttp.ClientSession() as session:
        # Prepare the POST request with the image data

        data = aiohttp.FormData()

        for i, image in enumerate(images):
            _, encoded_image = cv2.imencode(".jpg", image)
            image_bytes = encoded_image.tobytes()
            data.add_field("file", image_bytes, filename=f"image_{i}.jpg", content_type="image/jpeg")

        data.add_field('json_data', json.dumps(json_data), content_type='application/json')

        async with session.post(url, data=data) as response:
            response_json = await response.json()
            if response.status == 200:
                return response_json
            else:
                error = response_json['detail']
                print(f"Failed to send POST request. Status code: {response.status}. Error message: {error}")
                return




async def send_request_to_detector(image: np.ndarray,
                       url = "http://localhost:9029/api/detect-cars"
                       ):
    async with aiohttp.ClientSession() as session:
        # Prepare the POST request with the image data

        data = aiohttp.FormData()

        _, image = cv2.imencode(".jpg", image)
        image = image.tobytes()

        data.add_field("file", image, filename="image.jpg", content_type="image/jpeg")


        async with session.post(url,
                                data=data
                                ) as response:
            # response:
            answer = await response.json()
            if response.status == 200:
                return answer
            else:
                error = answer['detail']
                logger.log(level=4, msg=f"ERROR: {error}")
                return



async def send_request_to_segmentator(image: np.ndarray,
                                      json_data: dict,
                                   url = "http://localhost:9029/api/detect-cars"
                                   ):
    async with aiohttp.ClientSession() as session:
        # Prepare the POST request with the image data

        data = aiohttp.FormData()
        _, image = cv2.imencode(".jpg", image)
        image = image.tobytes()
        data.add_field("file", image, filename="image.jpg", content_type="image/jpeg")

        data.add_field('json_data', json.dumps(json_data), content_type='application/json')

        async with session.post(url,
                                data=data
                                ) as response:
            # response:
            answer = await response.json()
            if response.status == 200:
                return answer
            else:
                error = answer['detail']
                logger.log(level=4, msg=f"ERROR: {error}")
                return None




async def stream_video(video_path: os.path, total=100, request_to_url: str = "http://127.0.0.1"):
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ret, frame = cap.read()

        i += 1

        if i > total:
            # We skip N frames as cars don't move so fast
            break
        if not ret:
            break
        if 5 and i % 5 != 0:
            # We skip N frames as cars don't move so fast
            continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        await show_progress(current=i, total=total, item='frame')



        detector_answer = await send_request_to_detector(frame, url=f"{request_to_url}:9029/api/detect-cars")
        cars: list = detector_answer['data']

        if len(cars) == 0:
            # EMPTY
            logger.log(level=logger.level, msg=f"НЕ ОБНАРУЖЕНЫ МАШИНЫ В КАДРЕ")
            continue

        segmentator_answer = await send_request_to_segmentator(
            image=frame, json_data={"dots": cars},
            url=f"{request_to_url}:9029/api/extract-cars"
        )


        if segmentator_answer is not None:
            cars: list = segmentator_answer['data']

            classifier_answer = await send_request_to_classifier(
                image=frame, json_data={"dots": cars},
                url=f"{request_to_url}:9029/api/get-car-model"
            )
            labels = classifier_answer['data']
            logger.log(level=logger.level, msg=f"RESPONSE: {labels}")
        else:
            labels = ["Unknown"] * len(cars)
        # continue

        for i_file, car in enumerate(detector_answer['data']):
            box = car['box']
            box = np.array(box).astype(np.int32)
            car: np.ndarray = draw_rectangle(frame, bbox=box, label=labels[i_file])
            if car is None:
                continue
            await show_frame(car, windom_name='CAR {}'.format(i_file))
        # await asyncio.sleep(1)

    cap.release()
    cv2.destroyAllWindows()


async def main(run_on_localhost=True, url=None,):
    assert run_on_localhost or url
    if run_on_localhost:
        await asyncio.gather(
            run_web(public=False),
            stream_video(video_path ='../video/Гидростроителей.avi', total=3000, ),
        )
    else:
        await stream_video(video_path ='../video/Гидростроителей.avi', total=3000, request_to_url=url)


if __name__ == "__main__":
    asyncio.run(main(run_on_localhost=True, url="http://109.188.135.85",))




