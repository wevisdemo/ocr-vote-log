from typing import List
from src.detector import Detector
import easyocr
import numpy as np
import cv2
import torch
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image

class TableDetector:
  def __init__(self):
    self.feature_extractor = DetrFeatureExtractor.from_pretrained(
    "napatswift/paliament-vote-table-detection")
    self.model = DetrForObjectDetection.from_pretrained(
    "napatswift/paliament-vote-table-detection")

  def __call__(self, image):
    inputs = self.feature_extractor(images=image, return_tensors="pt")
    outputs = self.model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = self.feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

    return results["boxes"][results['scores'].argmax()]


class Recognizer:
    def __init__(self, reader: easyocr.Reader, images: np.ndarray) -> None:
        self.images = images
        self.detector = Detector(images)
        self.reader: easyocr.Reader = reader
        self.tab_detector = TableDetector()

    def get_text_image_list(self, image, text_bbox) -> List[np.ndarray]:
        
        text_images: List = list()
        for x, y, w, h in text_bbox:
            ti = image[y:y+h, x:x+w]
            text_images.append(ti)

        return text_images

    def get_column(self, image, bbox_list):
        _table = (0, 0, image.shape[1], image.shape[0])
        columns = self.detector.column_markers(
            img_width=image.shape[1], boxes=bbox_list, table=_table)
        middle_point = bbox_list[:, 0] + (bbox_list[:, 2] / 2)
        col_map = np.zeros(middle_point.shape)
        col_map.fill(-1)

        for i, col in enumerate(columns):
          x, y, w, h = col
          is_in = (middle_point >= x) & (middle_point <= x+w)
          col_map[is_in] = i
        
        return col_map

    def get_text(self, image: np.ndarray):
        M=15 # margin
        
        # find table region
        pil_img = Image.fromarray(image)
        tab = self.tab_detector(pil_img).tolist()
        x0,y0,x1,y1 = [int(x) for x in tab]
        cropped = image[y0-M:y1+M, x0-M:x1+M]
        
        bbox: np.ndarray = self.detector.detect_text(cropped)
        
        gray_image = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)

        filtered_bbox: np.ndarray = bbox[self.detector.filter(bbox)]
        columns = self.get_column(image, filtered_bbox)

        text_images = self.get_text_image_list(image=gray_image, text_bbox=filtered_bbox)
        
        # separate columns
        text_list: List = list()
        text_colum = [list() for x in range(int(columns.max())+1)]
        for img_txt, col_idx in zip(text_images, columns):
            recog_output = self.reader.recognize(img_txt)
            text = recog_output[0][1]
            text_colum[int(col_idx)].append(text)
            text_list.append(text)

        return text_colum

    def recognize(self):
        text_list = []
        for image in self.images:
            text = self.get_text(image)
            text_list.append(text)
        
        return text_list