from curses import COLOR_WHITE
import cv2
import numpy as np


class Detector:
    def __init__(self, images: np.array) -> None:
        self.images = images

    def detect(self):
        for image in self.images:
            self.process_image(image)
        pass

    def process_image(self, image):
        boxes = self.detect_text(image)

        boxes.sort(key=lambda x: x[0])
        boxes = np.array(boxes)

        boxes = boxes[filter(boxes)]

        line_nums = self.line()

        row_spans = self.row_span_in_page(line_nums)

        table = self.outer_box(row_spans)

        column_spans = self.column_markers(table)
        return

    def detect_column(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        hist_avg = None
        for im in self.images:
            gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            th, threshed = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            dilated = cv2.dilate(threshed, kernel, iterations=3)

            if hist_avg is None:
                hist_avg = cv2.reduce(dilated, 0, cv2.REDUCE_AVG)
            else:
                hist_avg = (
                    hist_avg + cv2.reduce(dilated, 0, cv2.REDUCE_AVG)) / 2

        hist_avg = hist_avg.reshape(-1)

        lined = im.copy()
        hist_b = hist_avg < hist_avg.max() - hist_avg.std()
        cols1 = np.argwhere(hist_b[1:] & (
            hist_b[1:] ^ hist_b[:-1])).reshape(-1)
        cols2 = np.argwhere(
            hist_b[:-1] & (hist_b[1:] ^ hist_b[:-1])).reshape(-1)
        cols = (cols1 + ((cols1 - cols2) * 0.8)).astype(int)

        return cols

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
        rects.sort(key=lambda rect: rect[1])

        return rects

    def row_span_in_page(self, line_nums):
        lnums = np.unique(line_nums)  # unique line number
        boxes_in_lines = [self.boxes[line_nums == i] for i in lnums]
        row_spans = np.array([self.outer_box(b) for b in boxes_in_lines])
        # thred = (row_spans[:, 2].mean() - row_spans.std()/2)
        # row_indexes = np.where(row_spans[:, 2] > thred)
        # row_spans = row_spans[row_indexes]
        return row_spans

    def column_markers(self, image, boxes, table):
        empty_image = np.zeros(image.shape)
        empty_image.fill(255)
        x_hist = np.zeros(image.shape[1])
        tx, ty, tw, th = table
        for x, y, w, h in boxes:
            if x >= tx and y >= ty and x+w <= tx+tw and y+h <= ty+th:
                x_hist[x:x+w] += h
            empty_image[y:y+h, x:x+w] = 0
        bxhist = x_hist.astype(bool)
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
        # sorting by x axis

        i = 0
        line_image = image.copy()
        while (box_line_num == -1).any():
            if box_line_num[i] != -1:
                i += 1
                continue
            center = sorted_boxes[:, 1] + (sorted_boxes[:, 3]/2)
            angle = np.arctan2(
                (center - center[i]), (sorted_boxes[:, 0] - sorted_boxes[i, 0]))

            is_in_line = (np.abs(angle) < 0.03)
            is_in_line_index = np.argwhere(is_in_line).reshape(-1)

            line_boxes = sorted_boxes[is_in_line_index]
            box_line_num[is_in_line] = i

            i += 1

        return box_line_num

    def filter(self, boxes: np.array):
        size_filter = (boxes[:, 2]*boxes[:, 3] > 300)
        whratio_filter = (boxes[:, 2]/boxes[:, 3] > 0.5)
        post_filter = (boxes[:, 0] != 0) & (boxes[:, 1] != 0)
        return size_filter & whratio_filter & post_filter

    def outer_box(self, boxes):
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
