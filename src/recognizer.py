from typing import List
from src.detector import Detector, TableDetector
import easyocr
import numpy as np
import cv2
from PIL import Image
from src.utils import onehot


class Recognizer:
    def __init__(self, reader: easyocr.Reader, images: np.ndarray) -> None:
        self.interested_chars = '- กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฤเแโใไะาุูิีืึั่้๊๋็์ำํฺฯๆ0123456789'
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

    def get_text(self, im_idx: int):
        assert isinstance(im_idx, int)
        image: np.ndarray = self.images[im_idx]

        # find table region
        pil_img = Image.fromarray(image)
        tab = self.tab_detector(pil_img).tolist()
        del pil_img

        bbox: np.ndarray = self.detector.detect_text(image)

        filtered_bbox: np.ndarray = bbox[self._filter(bbox, tab)]
        bbox_list: List = filtered_bbox.tolist()
        bbox_list.sort(key=lambda x: x[0])
        filtered_bbox = np.array(bbox_list)
        del bbox_list

        # draw y hist
        y_hist = np.zeros(image.shape[0])
        for x, y, w, h in filtered_bbox:
            y_hist[y:y+h] += w
        y_hist = y_hist != 0

        # find columns
        columns = self.get_column(image, filtered_bbox)

        # get image list
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        text_images = self.get_text_image_list(
            image=gray_image, text_bbox=filtered_bbox)
        text_image_arr = np.array(text_images, dtype=object)

        included = np.zeros(filtered_bbox[:, 1].shape, dtype=bool)

        text_lines = []
        prev_y = 1

        # DEV
        draw_line_img = image.copy()
        draw_line_img = self.__dev_draw_rect(draw_line_img, filtered_bbox)
        ###

        for ypos, ybool in enumerate(y_hist):
            if not ybool and prev_y:
                # find boxex that are in line
                line_b = (filtered_bbox[:, 1] < ypos) & (~included)
                if line_b.sum() == 0:
                    continue

                draw_line_img = self.__dev_draw_y_hline(draw_line_img, ypos)

                line_text = self._text_line(
                    text_image_arr[line_b], columns[line_b])

                text_lines.append(line_text)
                included[line_b] = ypos
            prev_y = ybool

        cv2.imwrite('line_p{}.png'.format(im_idx), draw_line_img)

        return text_lines

    def __dev_draw_rect(self, image, bbox):
        for x, y, w, h in bbox:
            image = cv2.rectangle(image, (x, y), (x+w, y+h), 255, 1)

        return image

    def __dev_draw_y_hline(self, image, ypos):
        image = cv2.line(image, (0, ypos), (image.shape[1], ypos), 255, 1)
        
        return image

    def _text_line(self, text_images, text_cols):
        line_text = [list() for _ in range(int(text_cols.max())+1)]
        for img_txt, col_txt in zip(text_images, text_cols):
            recog_output = self.reader.recognize(
                img_txt, allowlist=list(self.interested_chars))
            text = recog_output[0][1]
            col_idx = int(col_txt)
            line_text[col_idx].append(text)

        return line_text

    def recognize(self):
        text_list = []
        img_num = len(self.images)
        for i in range(img_num):
            text = self.get_text(i)
            text_list.append(text)

        return text_list
