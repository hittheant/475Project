import os
from sklearn.model_selection import train_test_split
import torchvision
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np


def load_data(filepath, target_shape, faces=False):
    imgs = list(sorted(os.listdir(f"{filepath}/images/")))
    files = []
    for img in imgs:
        [filename, _] = img.split('.')
        files.append(filename)
    data = []
    labels = []

    for filename in files:
        if faces:
            [boxes, mask_l] = load_image_data(filepath, f'{filename}.xml')
            data.extend(load_cropped_images(filepath, f'{filename}.png', boxes))
            labels.extend(mask_l.tolist())
        else:
            data.append(load_image(filepath, f'{filename}.png', target_shape))
            labels.append(load_image_data(filepath, f'{filename}.xml'))

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)

    return [x_train, x_test, y_train, y_test]


def load_image_data(filepath, filename):
    f = open(f'{filepath}/annotations/{filename}')
    data = f.read()
    soup = BeautifulSoup(data, 'html.parser')
    objects = soup.find_all('object')

    y = []
    boxes = []
    for obj in objects:
        xmin = int(obj.find('xmin').text)
        ymin = int(obj.find('ymin').text)
        xmax = int(obj.find('xmax').text)
        ymax = int(obj.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])

        if obj.find('name').text == "with_mask":
            y.append(1)
        elif obj.find('name').text == "without_mask":
            y.append(0)
        else:
            y.append(2)

    boxes = np.array(boxes)
    y = np.array(y)
    return [boxes, y]


def load_image(filepath, filename, target_shape):
    img = Image.open(f'{filepath}/images/{filename}')
    transform = torchvision.transforms.Resize(target_shape)
    img = transform(img)
    return np.asarray(img)


def load_cropped_images(filepath, filename, boxes):
    img = Image.open(f'{filepath}/images/{filename}')
    img = img.convert("RGB")
    transform = torchvision.transforms.Resize((30, 30))

    cropped_images = []
    for box in boxes:
        cropped_img = img.crop(box)
        cropped_images.append(np.asarray(transform(cropped_img)))

    return cropped_images


if __name__ == '__main__':
    data, labels = load_data('./archive', (600, 600), faces=True)
