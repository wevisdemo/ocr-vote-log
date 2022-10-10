from curses import COLOR_WHITE
from os import listdir
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class Detector:
    def __init__(self, images: np.ndarray):
        self.images = images

    def detect(self):
        detected: Dict = dict()

        text_lines: List = list()
        image_ids: List = list()
        boundering_boxes: List = list()
        for i, image in enumerate(self.images):
            data = self.process_image(image)
            text_lines.append(data['row'])
            boundering_boxes.append(data['bbox'])
            image_ids += ([i] * data['row'].shape[0])
        
        detected['line'] = np.concatenate(text_lines)
        detected['imaeg_id'] = np.array(image_ids)
        detected['bbox'] = np.concatenate(boundering_boxes)
        return detected

    def find_table_start_position(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        headers = self.table_header(gray_image)
        if len(headers) >= 2:
            header_start = np.mean(headers, axis=0)[1]
        else:
            header_start = headers[0][1]
        return header_start


    def process_image(self, image) -> Dict[str, np.ndarray]:
        boxes = self.detect_text(image)

        filterer = self.filter(boxes)
        boxes = boxes[filterer]

        line_nums = self.line(image, boxes)

        row_spans = self.row_span_in_page(boxes, line_nums)

        return {
            'row': row_spans,
            'bbox': boxes,
        }


    def detect_text(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blured = cv2.GaussianBlur(gray, (1, 9), 0)
        th, threshed = cv2.threshold(
            blured, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        threshed = cv2.dilate(threshed, kernel)
        # hist = cv2.reduce(threshed, 1, cv2.REDUCE_AVG).reshape(-1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        dilated = cv2.dilate(threshed, kernel)
        contours, hier = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rects = [cv2.boundingRect(c) for c in contours]
        rects.sort(key=lambda x: x[0])
        rects = np.array(rects)

        return rects

    def row_span_in_page(self, boxes, line_nums):
        lnums = np.unique(line_nums)  # unique line number
        boxes_in_lines = [boxes[line_nums == i] for i in lnums]
        row_spans = np.array([self.outer_box(b) for b in boxes_in_lines])
        # thred = (row_spans[:, 2].mean() - row_spans.std()/2)
        # row_indexes = np.where(row_spans[:, 2] > thred)
        # row_spans = row_spans[row_indexes]
        return row_spans

    def column_markers(self, img_width: int, boxes: np.ndarray, table: np.ndarray):
        x_hist = np.zeros(img_width)
        tx, ty, tw, th = [0] * 4
        tx, ty, tw, th = table
        for x, y, w, h in boxes:
            if x >= tx and y >= ty and x+w <= tx+tw and y+h <= ty+th:
                x_hist[x:x+w] += h
        bxhist = x_hist > x_hist.mean()
        bxor = np.logical_xor(bxhist[1:], bxhist[:-1])
        col_marks = np.argwhere(bxor)
        if col_marks.size % 2:
            col_marks = np.append(col_marks, [table[1]+table[3]])
        col_marks = col_marks.reshape(col_marks.size//2, 2)
        columns = []

        for x0, x1 in col_marks:
            columns.append((x0, table[1], x1-x0, table[3]))
        columns = np.array(columns)
        return columns

    def line(self, image: np.array, sorted_boxes: np.array) -> np.array:
        # create -1 matrix
        box_line_num = np.zeros(len(sorted_boxes))
        box_line_num.fill(-1)

        # for counting
        i = 0
        while (box_line_num == -1).any():
            if box_line_num[i] != -1:
                i += 1
                continue
            center = sorted_boxes[:, 1] + (sorted_boxes[:, 3]/2)
            angle = np.arctan2(
                (center - center[i]), (sorted_boxes[:, 0] - sorted_boxes[i, 0]))

            is_in_line = (np.abs(angle) < 0.03) # angle threshold

            box_line_num[is_in_line] = i

            i += 1

        return box_line_num
    
    def table_header(self, image):
        TEMPL_ROOT_DIR = 'src/patterns'
        TEMPLS = listdir(TEMPL_ROOT_DIR)
        header_locs = list()
        for template_fpath in TEMPLS:
            template_fpath = TEMPL_ROOT_DIR + '/' + template_fpath
            template = cv2.imread(template_fpath)
            template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
            res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.8:
                header_locs.append(max_loc)
        return header_locs


    def filter(self, boxes: np.array) -> np.ndarray:
        size_filter = (boxes[:, 2]*boxes[:, 3] > 300)
        whratio_filter = (boxes[:, 2]/boxes[:, 3] > 0.5)
        post_filter = (boxes[:, 0] != 0) & (boxes[:, 1] != 0)
        return size_filter & whratio_filter & post_filter

    def outer_box(self, boxes) -> Tuple:
        x0 = boxes[:, 0]
        x1 = (x0 + boxes[:, 2]).max()
        x0 = x0.min()
        y0 = boxes[:, 1]
        y1 = (y0 + boxes[:, 3]).max()
        y0 = y0.min()
        return (x0, y0, x1-x0, y1-y0)

    def detect_column_header(self, image):
        COL_HEADER_IM = 'lamdabti.png'
        header = cv2.imread(COL_HEADER_IM)
        result = cv2.matchTemplate(header, image, cv2.TM_SQDIFF_NORMED)
        _, max_val, mnLoc, _ = cv2.minMaxLoc(result)
        if max_val < 0.8: # threshold
            return
        return mnLoc
