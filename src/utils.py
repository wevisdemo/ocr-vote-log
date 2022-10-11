import cv2
import numpy as np
from difflib import SequenceMatcher


def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)


def similar(a, b):
    return SequenceMatcher(lambda x: x in ' ', a, b).ratio()


def fix_vote(vote: str):
    VOTE_TYPES = ['เห็นด้วย', 'ไม่เห็นด้วย',
                  'งดออกเสียง', 'ไม่ลงคะแนนเสียง', '-']
    '''
    1 = เห็นด้วย,
    2 = ไม่เห็นด้วย,
    3 = งดออกเสียง,
    4 = ไม่ลงคะแนนเสียง, 
    5 = ไม่เข้าร่วมประชุม,
    \- = ไม่ใช่วาระการประชุม
    '''
    if not isinstance(vote, str):
        return '-'

    # common ocr error
    vote = vote.replace('.ท็น', 'เห็น')\
               .replace('เท็น', 'เห็น')\
               .replace('เห็นด้าย', 'เห็นด้วย')

    if vote in VOTE_TYPES:
        return vote
    p = [similar(vote, v) for v in VOTE_TYPES]
    i = np.argmax(p)
    if p[i] < .6:
        # print(vote)
        return '-'
    return VOTE_TYPES[i]


def fix_party(p, ps):
    if not p:
        return p
    if p in ps:
        return p
    sim_l = []
    for x in ps:
        sim = similar(p, str(x))
        sim_l.append(sim)
    if max(sim_l) > .7:
        return ps[np.argmax(sim_l).reshape(-1)[0]]
    return p


def outer_box(boxes):
    x0 = boxes[:, 0]
    x1 = (x0 + boxes[:, 2]).max()
    x0 = x0.min()
    y0 = boxes[:, 1]
    y1 = (y0 + boxes[:, 3]).max()
    y0 = y0.min()
    return (x0, y0, x1-x0, y1-y0)


def draw_boxes(boxes, image, self=None):
    canvas = image.copy()
    for x, y, w, h in boxes:
        canvas = cv2.rectangle(canvas, (x, y), (x+w, y+h), 255, 1)
    if self is not None:
        x, y, w, h = self
        canvas = cv2.rectangle(canvas, (x, y), (x+w, y+h), 1, 2)
    x0, y0, w, h = outer_box(boxes)
    canvas = cv2.rectangle(canvas, (x0, y0), (x0+w, y0+h), (100, 100, 1000), 1)
    return canvas

def onehot(arr):
    ''' Returns a 1-hot encoding array of arr
    With no zero groups, returns sorted labels
    '''
    arr_copy = np.copy(arr)
    unique_arr = np.unique(arr_copy)
    one_hot = (arr_copy.reshape(-1,1) == unique_arr.reshape(1,-1)).astype(np.int32)
    return one_hot
