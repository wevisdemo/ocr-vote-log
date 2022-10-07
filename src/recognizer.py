import cv2
import tqdm

def parse_text(page_list, reader, columns):
  rects_in_lines = []
  log = []
  for image in tqdm(page_list):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blured = cv2.GaussianBlur(gray, (9,9), 0)
    th, threshed = cv2.threshold(blured, 200, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    threshed = cv2.dilate(threshed, kernel)
    hist = cv2.reduce(threshed, 1, cv2.REDUCE_AVG).reshape(-1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    dilated = cv2.dilate(threshed, kernel)
    contours, hier  = cv2.findContours(
          dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rects = [cv2.boundingRect(c) for c in contours]
    rects.sort(key=lambda rect: rect[1])

    h, w = image.shape[:2]
    stack_y = []
    for ii in range(h-1):
      if (hist[ii] and not hist[ii+1]):
        stack_y.append(ii)
      elif (not hist[ii] and hist[ii+1]):
        stack_y.append(ii)

    line_num = len(stack_y)//2
    while(stack_y):
      rects_in_col = [[] for _ in range(len(columns)+1)]
      line_y2 = stack_y.pop(0)
      line_y1 = stack_y.pop(0)
      line_rects = []
      while rects:
        rect = rects[0]
        y_mid_point = rect[1] + rect[3]//2
        x_mid_point = rect[0] + rect[2]//2
        if line_y2 < y_mid_point < line_y1:
          rect = rects.pop(0)
          line_rects.append(rect)
        else:
          break
      line_rects.sort(key=lambda x: x[0])

      for x, y, w, h in line_rects:
        col = (x + (w/2) > columns).sum()
        text = reader.recognize(image[y:y+h, x:x+w])[0][1]
        if col >= len(rects_in_col):
          log.append((col, text))
          continue
        rects_in_col[col].append(text)
      rects_in_lines.append(rects_in_col)
  return rects_in_lines