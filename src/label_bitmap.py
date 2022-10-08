import json
import numpy as np
import cv2


class COCODataLoader:
    id2label = {
        0: 'table',
        1: 'table column',
        2: 'table row',
        3: 'table column header',
        4: 'table projected row header',
        5: 'table spanning cell',
        6: 'text',
    }

    def __init__(self, cocofpath) -> None:
        with open(cocofpath, 'r') as fp:
            self.coco = json.load(fp)
        self.images = self.coco['images']
        self.categories = self.coco['categories']
        self.annotations = self.coco['annotations']

        self.process_bitmap()

    def process_bitmap(self):
        self.labels = {}

        for image in self.images:
            id = image['id']
            empt_image = np.zeros((image['height'], image['width']))
            self.labels[id] = empt_image

        for ann in self.annotations:
            image_id = ann['image_id']
            bitmap = self.labels[image_id]
            x, y, w, h = ann['bbox']
            bitmap[y:y+h, x:x+w] += ann['category_id']

    def __getitem__(self, imid):
        pass


if __name__ == '__main__':
    dl = COCODataLoader(
        'ds/project-4-at-2022-10-08-03-27-552b26e7/result.json')
    cv2.imwrite('bitmap.jpg', dl.labels[2]*10)
