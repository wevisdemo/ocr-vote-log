import os
from typing import List
import requests
import numpy as np
import cv2
import fitz

def download(url: str) -> str:
    filename = url.rsplit('/', maxsplit=1)[1]
    if not os.path.exists(filename):
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(
                f'status code {response.status_code} (expected 200) [{url}]')
        with open(filename, 'wb') as fp:
            fp.write(response.content)
    return filename

def convert_to_image(file_path: str) -> np.ndarray:
    if not os.path.exists(file_path):
        return np.array([])
    doc = fitz.open(file_path)

    # find maximum width and height
    max_dim: List = [0, 0]
    for page in doc:
        pix_map = page.get_pixmap()
        if pix_map.height > max_dim[0]:
            max_dim[0] = pix_map.height
        if pix_map.width > max_dim[1]:
            max_dim[1] = pix_map.width

    images: List = list()
    for page in doc:
        pix_map = page.get_pixmap()
        height, width = pix_map.height, pix_map.width
        image_mat = np.frombuffer(pix_map.samples_mv, dtype=np.uint8).reshape((height, width, 3))

        # add padding
        padded = cv2.copyMakeBorder(image_mat, 0, max_dim[0]-height, 0, max_dim[1]-width, cv2.BORDER_CONSTANT, value=255)

        images.append(padded)
    
    # convert to single array
    image_mats = np.array(images)
    return image_mats
    
