from typing import List
from src.detector import Detector, TableDetector
import easyocr
import numpy as np
import cv2
from PIL import Image
from src.utils import onehot


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
        M = 10
        _table = (0, 0, image.shape[1], image.shape[0])
        columns = self.detector.column_markers(
            img_width=image.shape[1], boxes=bbox_list, table=_table)
        middle_point = bbox_list[:, 0] + (bbox_list[:, 2] / 2)
        col_map = np.zeros(middle_point.shape)
        col_map.fill(-1)

        for i, col in enumerate(columns):
            x, y, w, h = col
            is_in = (middle_point >= x) & (middle_point <= x+w+M)
            col_map[is_in] = i

        return col_map

    def _filter(self, bbox, table):
        x0, y0, x1, y1 = table
        x_pos = bbox[:, 0] + bbox[:, 2]/2
        y_pos = bbox[:, 1] + bbox[:, 3]/2
        return self.detector.filter(bbox) & (
            (x_pos > x0) & (x_pos < x1) & (y_pos > y0) & (y_pos < y1))

    def get_text(self, image: np.ndarray):
        M = 15  # margin

        # find table region
        pil_img = Image.fromarray(image)
        tab = self.tab_detector(pil_img).tolist()
        x0, y0, x1, y1 = [int(x) for x in tab]

        bbox: np.ndarray = self.detector.detect_text(image)

        filtered_bbox: np.ndarray = bbox[self._filter(bbox, tab)]
        bbox_list: List = filtered_bbox.tolist()
        bbox_list.sort(key=lambda x: x[0])
        filtered_bbox = np.array(bbox_list)

        # draw y hist
        y_hist = np.zeros(image.shape[0])
        for x, y, w, h in filtered_bbox:
            y_hist[y:y+h] += w
        y_hist = y_hist > y_hist.mean() - y_hist.std()

        # find columns
        columns = self.get_column(image, filtered_bbox)

        # get image list
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        text_images = self.get_text_image_list(
            image=gray_image, text_bbox=filtered_bbox)
        text_image_arr = np.array(text_images)

        col_num = np.unique(columns)
        text_list: List = list()

        included = np.zeros(filtered_bbox[:, 1].shape, dtype=bool)

        text_lines = []
        prev_y = 1
        for i, y in enumerate(y_hist):
            if y and not prev_y:
                # find boxex that are in line
                line_b = (filtered_bbox[:, 1] < i) & (~included)
                if line_b.sum() == 0:
                    continue
                text_cols = columns[line_b]
                text_imgs = text_image_arr[line_b]
                line_text = self._text_line(text_imgs, text_cols)
                text_lines.append(line_text)
                included[line_b] = i
            prev_y = y

        return text_lines

    def _text_line(self, text_images, text_cols):
        line_text = [list() for _ in range(int(text_cols.max())+1)]
        for img_txt, col_txt in zip(text_images, text_cols):
            recog_output = self.reader.recognize(img_txt)
            text = recog_output[0][1]
            col_idx = int(col_txt)
            line_text[col_idx].append(text)
        return line_text

    def _line_order(self, bboxes: np.array, lines: np.ndarray) -> np.ndarray:
        one_hot = onehot(lines)
        vert_sum = one_hot.T@bboxes  # summation of each line
        box_count = one_hot.sum(axis=0)  # number of boxes each line
        box_mid = (vert_sum[:, 1])
        avg = (box_mid/box_count)
        return avg.argsort()

    def recognize(self):
        text_list = []
        for image in self.images:
            text = self.get_text(image)
            text_list.append(text)

        return text_list
