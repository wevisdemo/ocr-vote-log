from typing import List
from src.detector import Detector
import easyocr
import numpy as np
import cv2


class Recognizer:
    def __init__(self, reader: easyocr.Reader, images: np.ndarray) -> None:
        self.images = images
        self.detector = Detector(images)
        self.reader: easyocr.Reader = reader

    def get_text_image_list(self, image, text_bbox) -> List[np.ndarray]:
        filtered_bbox: np.ndarray = text_bbox[self.detector.filter(text_bbox)]

        text_images: List = list()
        for x, y, w, h in filtered_bbox:
            ti = image[y:y+h, x:x+w]
            text_images.append(ti)

        return text_images

    def get_text(self, image: np.ndarray):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        bbox: np.ndarray = self.detector.detect_text(image)
        text_images = self.get_text_image_list(image=gray_image, text_bbox=bbox)
        text_list: List = list()

        for img_txt in text_images:
            recog_output = self.reader.recognize(img_txt)
            text = recog_output[0][1]
            text_list.append(text)

        return text_list

    def recognize(self):
        text_list = []
        for image in self.images:
            text = self.get_text(image)
            text_list.append(text)
        
        return text_list
