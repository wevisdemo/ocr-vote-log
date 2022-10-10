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
  
  def test_column_count(self):
    detector = Detector([])
    image, boxes = self._get_boxes(detector, 'test/test_case/table-cropped.png')
    table = (0, 0, image.shape[1], image.shape[0])
    columns = detector.column_markers(img_width=image.shape[1], boxes=boxes, table=table)
    self.assertEqual(columns.shape[0], 5)


  def test_detect_table_header(self,):
    root = 'test/pdf'
    image_paths = [os.path.join(root, fname) for fname in listdir(root) if fname.endswith('.png')]
    image_paths = image_paths[:100]
    images = [cv2.imread(fp) for fp in image_paths][:100]
    gray_images = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images]
    detector = Detector([])
    for fp, image in zip(image_paths, gray_images):
      if image is None: continue
      with self.subTest(fp):
        res = detector.table_header(image)
        self.assertGreaterEqual(len(res), 1, f'file: {fp}')