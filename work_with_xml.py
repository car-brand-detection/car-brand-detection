import os
import re
import time
import aiohttp
import asyncio
import requests
import xml.etree.ElementTree as ET


# All tags:
# {'link', 'plate_number_image_url', 'plate_id', 'plate_title', 'tags', 'plate_region', 'fon_id', 'fon_title', 'model', 'photo_url', 'plate_number', 'country', 'model2', 'car'}

car_pattern = r"([a-zA-Z0-9а-яА-Я]+)"


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
    response = requests.get(url)
    response.raise_for_status()

    with open(save_path, 'wb') as file:
        file.write(response.content)

    # print(f"Image downloaded at: {save_path}")


async def download_image_asynchronously(url, save_path):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()
            with open(save_path, 'wb') as file:
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    file.write(chunk)

    # print(f"Image downloaded asynchronously at: {save_path}")


TEST_URL = "https://cdn.pixabay.com/photo/2023/05/15/09/18/iceberg-7994536_1280.jpg"


async def parse_xml(xml_files_path: str, save_result_to: str, test_mode=True, asynchronously = False):
    """ Function to parse all XML files in folder and create all folders according to model names"""
    save_result_to += "/"
    xml_files = os.listdir(xml_files_path)
    for file_n, file_path in enumerate(xml_files[:2]):
        with open(xml_files_path + file_path, 'r', encoding='utf-8') as file:
            tree = ET.parse(file)
        root = tree.getroot()

        # Access elements and attributes in the XML file
        for n, child in enumerate(root[:]):

            car = child.find("car").text
            if not car or len(car) < 2:
                continue
            photo_url = child.find("photo_url").text

            image_name = get_car_image_name(photo_url)

            url = TEST_URL if test_mode else photo_url


            model = child.find("model").text
            model2 = child.find("model2").text


            tags = [car, model, model2]

            for tag_n, tag in enumerate(tags[:], start=0):
                tag = get_folder_name(tag)
                tags[tag_n] = remove_slash_and_other_trash(tag)

            picture_path = save_result_to + '/'.join(tags)
            try:
                os.makedirs(picture_path, exist_ok=True)

                if asynchronously:
                    await download_image_asynchronously(url=url, save_path=picture_path + '/' + image_name)
                else:
                    download_image(url=url, save_path=picture_path + '/' + image_name)

            except:
                raise
        print(f"File {file_n} is processed successfully.")




async def main():
    await parse_xml(xml_files_path="../50k/", save_result_to="Cars", test_mode=True, asynchronously=True)

# ##### For Jupyter, we already have a loop, so we can just await the fucntion:
# await download_image(url="", save_path="")

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    print(f"Code execution took {time.time() - start_time} seconds.")


