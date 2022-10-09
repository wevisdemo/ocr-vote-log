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

def convert_image(file_path: str) -> np.ndarray:
    if not os.path.exists(file_path):
        return np.array([])
    doc = fitz.open(file_path)


    images: List = list()
    max_w, max_h = (0, 0)
    
    zoom_mat = fitz.Matrix(3, 3)
    for page in doc:
        pix_map = page.get_pixmap(matrix=zoom_mat)
        height, width = pix_map.height, pix_map.width
        if max_h < height:
            max_h = height
        if max_w < width:
            max_w = width

        image_mat = np.frombuffer(pix_map.samples_mv, dtype=np.uint8).reshape((height, width, 3))
        images.append(image_mat)
    
    padded_images = []
    for image_mat in images:
        # add padding
        padded = cv2.copyMakeBorder(image_mat, 0, max_h-height, 0, max_w-width, cv2.BORDER_CONSTANT, value=255)
        padded_images.append(padded)

    # convert to single array
    image_mats = np.array(padded_images)
    return image_mats
    
