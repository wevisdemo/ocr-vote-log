from os import listdir
import os
import unittest
from src.detector import Detector
from src.utils import draw_boxes
import cv2
import numpy as np

class TestDetector(unittest.TestCase):

  def _get_boxes(self, detector, image_path: str):
    image = cv2.imread(image_path)
    boxes = detector.detect_text(image)
    boxes = boxes[detector.filter(boxes)]
    return image, boxes 

  def test_table_only_line_count(self):
    detector = Detector([])
    image, boxes = self._get_boxes(detector, 'test/test_case/26_lines.png')
    line_num_map = detector.line(image, boxes)
    drawed = draw_boxes(boxes, image.copy())
    cv2.imwrite('test/test_output/test_table_only_line_count.png', drawed)
    self.assertEqual(26, np.unique(line_num_map).size)

  def test_page_line_count(self):
    detector = Detector([])
    image, boxes = self._get_boxes(detector, 'test/test_case/37_lines.png')
    line_num_map = detector.line(image, boxes)
    drawed = draw_boxes(boxes, image.copy())
    cv2.imwrite('test/test_output/test_page_line_count.png', drawed)
    self.assertEqual(37, np.unique(line_num_map).size)

  def test_detect_table_header(self,):
    root = 'test/pdf'
    image_paths = [os.path.join(root, fname) for fname in listdir(root) if fname.endswith('.png')]
    image_paths = image_paths[:100]
    images = [cv2.imread(fp, 0) for fp in image_paths][:100]
    detector = Detector([])
    for fp, image in zip(image_paths, images):
      if image is None: continue
      with self.subTest(f'test find header in {fp}'):
        res = detector.table_header(image)
        self.assertGreaterEqual(len(res), 1, f'file: {fp}')