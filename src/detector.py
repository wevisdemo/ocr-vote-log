import cv2
import numpy as np
from src.utils import noise_removal

def detect_column(images):
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
  hist_avg = None
  for im in images:
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    noise_removed = noise_removal(gray)
    th, threshed = cv2.threshold(noise_removed, 200, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(threshed, kernel, iterations=3)

    if hist_avg is None:
      hist_avg = cv2.reduce(dilated, 0, cv2.REDUCE_AVG)
    else:
      hist_avg = (hist_avg + cv2.reduce(dilated, 0, cv2.REDUCE_AVG)) / 2

  hist_avg = hist_avg.reshape(-1)

  lined = im.copy()
  hist_b = hist_avg < hist_avg.max() - hist_avg.std()
  cols1 = np.argwhere(hist_b[1:] & (hist_b[1:] ^ hist_b[:-1])).reshape(-1)
  cols2 = np.argwhere(hist_b[:-1] & (hist_b[1:] ^ hist_b[:-1])).reshape(-1)
  cols = (cols1 + ((cols1 - cols2) * 0.8)).astype(int)

  for col in cols:
    cv2.line(lined, (col, 0), (col, 400), (0, 0, 255), 2)

  cv2.imwrite('lines.jpg', lined)
  return cols

