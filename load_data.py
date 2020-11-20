import os

import torchvision
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np


def load_data(filepath, target_shape):
    imgs = list(sorted(os.listdir(f"{filepath}/images/")))
    num_examples = len(imgs)
    files = []
    for img in imgs:
        [filename, _] = img.split('.')
        files.append(filename)
    data = []
    labels = []
    for i, filename in enumerate(files):
        data.append(load_image(filepath, f'{filename}.png', target_shape))
        labels.append(load_image_data(filepath, f'{filename}.xml'))


def load_image_data(filepath, filename):
    f = open(f'{filepath}/{filename}')
    data = f.read()
    soup = BeautifulSoup(data, 'xml')
    objects = soup.find_all('object')

    num_objs = len(objects)
    labels = []
    boxes = []
    for obj in objects:
        xmin = int(obj.find('xmin').text)
        ymin = int(obj.find('ymin').text)
        xmax = int(obj.find('xmax').text)
        ymax = int(obj.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])

        if obj.find('name').text == "with_mask":
            labels.append(1)
        elif obj.find('name').text == "without_mask":
            labels.append(0)
        else:
            labels.append(2)

    boxes = np.array(boxes)
    labels = np.array(labels)
    return [boxes, labels]


def load_image(filepath, filename, target_shape):
    img = Image.open(f'{filepath}/images/{filename}')
    transform = torchvision.transforms.Resize(target_shape)
    img = transform(img)
    print(img.shape())
    return np.asarray(img)


if __name__ == '__main__':
    load_data('./archive', (600, 600))
