from typing import List
from src.detector import Detector
import easyocr
import numpy as np


class Recognizer:
    def __init__(self, reader: easyocr.Reader, images: np.ndarray) -> None:
        self.images = images
        self.detector = Detector(images)
        self.reader = reader


    def get_text_image_list(self, image) -> List[np.ndarray]:
        bbox: np.ndarray = self.detector.detect_text(image)
        filtered_bbox: np.ndarray = bbox[self.detector.filter(bbox)]
        
        text_images: List = list()
        for x, y, w, h in filtered_bbox:
            # crop image
            ti = image[y:y+h, x:x+w]
            text_images.append(ti)

        return text_images

    def get_text(self):
        # for activate reader
        dummy = np.zeros(self.images.shape, dtype=np.uint8)
        self.reader.readtext_batched()


