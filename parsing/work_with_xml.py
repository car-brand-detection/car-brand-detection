import os
import re
import ssl
import time
import asyncio

import urllib.request
import xml.etree.ElementTree as ET

import requests
import aiohttp
from tqdm import tqdm

# All tags:
# {'link', 'plate_кnumber_image_url', 'plate_id', 'plate_title', 'tags', 'plate_region', 'fon_id', 'fon_title', 'model', 'photo_url', 'plate_number', 'country', 'model2', 'car'}

car_pattern = r"([a-zA-Z0-9а-яА-Я]+)"
TIMEOUT = aiohttp.ClientTimeout(total=600)

def get_folder_name(generation: str) -> str:
    """
    Generation name looks like a mess. This func will extract only numeric info
    :param generation:
    :return: number or string like "2nd"
    """
    if generation is None: return '0'

    match = re.search(car_pattern, generation)
    if match:
        return match.group(1)

    try:
        int(generation)
        return generation
    except ValueError:
        gen = generation.split()[0]
        if gen[-1] in ',:.;':
            gen = gen[:-1]
        return gen

def get_car_image_name(url: str) -> str:
    """
    Function to get unique name for umage. All url are unique (as long as pictures are different), so why don't we use that
    """
    name = url.split('/')[-1]
    # name = name.split('.')[0]
    return name

def remove_slash_and_other_trash(model:str) -> str:
    """
    We don't need lots of nested folders!
    """
    if model:
        return model.replace("/", "_&_").replace(":", "-")
    else:
        return '0'

def download_image(url, save_path):
    response = requests.get(url, stream=True, verify=ssl.CERT_NONE)
    response.raise_for_status()

    with open(save_path, 'wb') as file:
        file.write(response.content)
    return True


from urllib.parse import urlparse

async def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and (
                "http" in result.scheme or "https" in result.scheme
        )
    except ValueError:
        return False



def get_connector_with_disabled_ssl():
    connector = aiohttp.TCPConnector()
    return connector

async def download_image_asynchronously(url, save_path, use_ssl: bool = False):

    connector = None if use_ssl else get_connector_with_disabled_ssl()

    async with aiohttp.ClientSession(trust_env=True,
                                     connector=connector,
                                     timeout=TIMEOUT
                                     ) as session:
        async with session.get(url, ssl=False, timeout=TIMEOUT) as response:
            if response.status == 200:
                with open(save_path, 'wb') as file:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        file.write(chunk)
                # print("Saved: {}".format(save_path))
                return True
            elif response.status == 404:
                print(f"The image from {url} WAS NOT saved to {save_path} due to BROKEN URL (404). Program goes further (running...)")
            elif response.status == 403:
                pass
                # print(f"ERROR 403: NO ACCESS!")
            elif response.status == 504:
                print(f"CODE 504: TIMEOUT ERROR!")
                # response.raise_for_status()
            else:
                print(f"UNEXPECTED ERROR! The image from {url} WAS NOT saved to {save_path} due to. HTTP ERROR CODE IS: {response.status}.\nIf the error persists, TERMINATE THE PROGRAM!!!")
                await asyncio.sleep(5)
                response.raise_for_status()
    # print(f"ERROR {response.status} WITH URL: {url}")
    return False



    # print(f"The image from {url} mage downloaded asynchronously at: {save_path}")


TEST_URL = "https://cdn.pixabay.com/photo/2023/05/15/09/18/iceberg-7994536_1280.jpg"



async def show_progress(current=0, total=30, size=None, item='file'):
    size = total if not size else size
    string_pointer = int(current / total * size)
    label = f"|{item} {current}|"
    label = f"{label:->{string_pointer + len(label)}}"
    label = f"{label:><{size - string_pointer + len(label)}}"
    label += f"|of {total}|"
    print("\n", label)

async def parse_xml(xml_files_path: str, save_result_to: str, test_mode=True, asynchronously=False, skip_exist=True,
                    divide_by='category', request_delay = 1e-9):
    """
         Function to parse all XML files in folder and create all folders according to model names or XML file names

    :param xml_files_path:
    :param save_result_to:
    :param test_mode:
    :param asynchronously:
    :param skip_exist:
    :param divide_by: what will we use as subfolders names: categories (
        e.g. models) or source files names. For second case use 'file_name'
    :return: None
    """
    assert divide_by in ['category', 'file_name']
    save_result_to += "/"
    xml_files = os.listdir(xml_files_path)
    number_of_iterations = len(xml_files)


    for file_n, file_path in enumerate(xml_files, start=1):
        with open(xml_files_path + file_path, 'r', encoding='utf-8') as file:
            tree = ET.parse(file)
        root = tree.getroot()
        await show_progress(current=file_n, total=number_of_iterations, size=30)

        total = 0
        # broken = 0
        exists = 0
        downloaded = 0
        failed = 0

        # Access elements and attributes in the XML file
        for n, child in enumerate(
                tqdm(root, desc='XML processing...'),
        ):

            car = child.find("car").text
            if not car or len(car) < 2:
                continue
            photo_url = child.find("photo_url").text

            image_name = get_car_image_name(photo_url)

            url = TEST_URL if test_mode else photo_url

            # if not await validate_url(url):
            #     broken += 1

            total += 1
            # if total and total % 100 == 0:
            #     break

            model = child.find("model").text
            # model2 = child.find("model2").text


            if divide_by == 'category':
                tags = [
                    car,
                    model,
                    # model2
                ]

                for tag_n, tag in enumerate(tags, start=0):
                    tag = get_folder_name(tag)
                    tags[tag_n] = remove_slash_and_other_trash(tag)

                picture_path = save_result_to + '/'.join(tags)
            else:
                picture_path = save_result_to + '/' + file_path.split(".")[0]

            os.makedirs(picture_path, exist_ok=True)

            end_path = picture_path + '/' + image_name
            if skip_exist:
                if image_name in os.listdir(picture_path):
                    # Means image already exists, we can skip
                    exists += 1
                    continue

            try:
                if asynchronously:
                    success = await download_image_asynchronously(url=url, save_path=end_path)

                else:
                    success = download_image(url=url, save_path=end_path)
            except (requests.HTTPError, aiohttp.ClientConnectorError):
                print(f"{end_path} wasn't downloaded due to HTTP Error.  Program STOPPED at this moment")
                raise
            except:
                raise

            if success:
                downloaded += 1
            else:
                failed += 1
                # await asyncio.sleep(0.001)

            await asyncio.sleep(request_delay)

            # if downloaded and downloaded % 10 == 0:
            #     pass
            #     print(f"{downloaded} of images were downloaded for current XML file.\n")

            # if total and total % 100 == 0:
            #     break

        if downloaded + exists > 0 and failed < downloaded + exists:
        # if failed == 0:
            print(
                f"File {file_n} ({file_path}) is processed successfully. For this file:\n"
                f"- {downloaded} картинок скачали.\n"
                f"- {exists} картинок уже скачаны (пропущены).\n"
                f"- {total} обработано в сумме.\n"
                # f"- {broken} поломанных ссылок.\n"
                f"- {failed} НЕ получилось скачать.\n"
            )
        else:
            print(f"\nFAILED: файл {file_path}; номер {file_n}.\n"
                  f"- {total} обработано в сумме.\n"
                  f"- {failed} НЕ получилось скачать.\n"
            )
            request_delay *= 10
            print(f"Увеличиваем задержку в {10} раз - до {request_delay}. Начинаем скачивание заново.")
            return await parse_xml(
                xml_files_path, save_result_to,
                test_mode, asynchronously, skip_exist,
                divide_by, request_delay
            )



async def main():
    # await parse_xml(xml_files_path="../data/XML/", save_result_to="../data/PPARSED_XML/", test_mode=False, asynchronously=True, skip_exist=True, divide_by='category')
    await parse_xml(xml_files_path="D:/DataSets/PM2/XML/", save_result_to="D:/DataSets/PM2/Images/", test_mode=False, asynchronously=True, skip_exist=True, divide_by='category')

# ##### For Jupyter, we already have a loop, so we can just await the fucntion:
# await download_image(url="", save_path="")

if __name__ == "__main__":
    start_time = time.time()
    policy = asyncio.WindowsSelectorEventLoopPolicy()
    asyncio.set_event_loop_policy(policy)

    asyncio.run(main())
    print(f"Code execution took {time.time() - start_time} seconds.")




#%%

#%%

#%%
