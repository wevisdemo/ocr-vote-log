import json
import cv2
import random

data = json.load(open('textdet-thvote/textdet_test.json'))
data = random.choice(data['data_list'])

img = cv2.imread('textdet-thvote/imgs/'+data['img_path'])
if img is None:
  raise FileExistsError('image file not found')
for text_reg in data['instances']:
  polygon = text_reg['polygon']
  polygon = [int(x) for x in polygon]
  for i in range(len(polygon)//2-1):
    ix0 = i*2
    ix1 = (i+1)*2
    cv2.line(img, [polygon[ix0], polygon[ix0+1]], [polygon[ix1], polygon[ix1+1]], (255, 0, 0))
  bbox = text_reg['bbox']
  bbox = [int(x) for x in bbox]
  cv2.rectangle(img, bbox[:2], bbox[2:], (0, 0, 255), 2)

cv2.imshow(data['img_path'], img)
cv2.waitKey(0)

