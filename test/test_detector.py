import unittest
from src.detector import Detector
from src.utils import draw_boxes
import cv2
import numpy as np

class TestDetector(unittest.TestCase):

  def _get_boxes(self, detector, image_path: str):
    image = cv2.imread(image_path)
    boxes = detector.detect_text(image)
    boxes.sort(key=lambda x: x[0])
    boxes = np.array(boxes)
    boxes = boxes[detector.filter(boxes)]
    return image, boxes 

  def test_table_only_line_count(self):
    detector = Detector([])
    image, boxes = self._get_boxes(detector, 'test/test_case/26_lines.png')
    line_num_map = detector.line(image, boxes)
    drawed = draw_boxes(boxes, image.copy())
    cv2.imwrite('test/test_output/test_table_only_line_count.png', drawed)
    self.assertEqual(26, np.unique(line_num_map).size)

  def test_paeg_line_count(self):
    detector = Detector([])
    image, boxes = self._get_boxes(detector, 'test/test_case/37_lines.png')
    line_num_map = detector.line(image, boxes)
    drawed = draw_boxes(boxes, image.copy())
    cv2.imwrite('test/test_output/test_page_line_count.png', drawed)
    self.assertEqual(37, np.unique(line_num_map).size)

  def test_columns_count(self):
    pass