import os
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import torchvision
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np

numpy_split_names = ['x_train.npy', 'x_test.npy', 'y_train.npy', 'y_test.npy']


if not os.path.exists('./results/'):
    os.makedirs('./results/')
if not os.path.exists('./archive/balanced_splits'):
    os.makedirs('./archive/balanced_splits')
if not os.path.exists('./archive/raw_splits'):
    os.makedirs('./archive/raw_splits')


def resample_data(x, y):
    x = np.array(x)
    y = np.array(y)
    majority = x[y == 1]
    minority = x[y == 2]
    middle = x[y == 0]
    target_count = len(minority) * 2

    ys = []
    for i in range(3):
        ys.append(np.ones(target_count) * i)
    ys = np.array(ys).flatten()

    minority = resample(minority, replace=True, n_samples=target_count, random_state=0)
    middle = resample(middle, replace=False, n_samples=target_count, random_state=0)
    majority = resample(majority, replace=False, n_samples=target_count, random_state=0)
    xs = np.concatenate((middle, majority, minority), axis=0)
    data = list(zip(xs, ys))
    np.random.shuffle(data)
    xs, ys = zip(*data)
    return xs, ys


def load_data(filepath, target_shape, faces=False):
    if os.path.exists(f'./archive/balanced_splits/x_train.npy'):
        balanced_splits = []
        for n in numpy_split_names:
            balanced_splits.append(np.load(f'./archive/balanced_splits/{n}'))
        return balanced_splits
    if not os.path.exists(f'./archive/raw_splits/x_train.npy'):
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
        splits = [x_train, x_test, y_train, y_test]
        for n, s in zip(numpy_split_names, splits):
            np.save(f'./archive/raw_splits/{n}', s)
    else:
        splits = []
        for n in numpy_split_names:
            splits.append(np.load(f'./archive/raw_splits/{n}'))
        (x_train, x_test, y_train, y_test) = splits

    x_train, y_train = resample_data(x_train, y_train)
    x_test, y_test = resample_data(x_test, y_test)
    splits = [x_train, x_test, y_train, y_test]
    for n, s in zip(numpy_split_names, splits):
        np.save(f'./archive/balanced_splits/{n}', s)

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
    splits = load_data('./archive', (600, 600), faces=True)
