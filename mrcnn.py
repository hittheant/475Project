import os
import sys
import random
import math
import re
import time
import cv2
import torchvision
from bs4 import BeautifulSoup
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath("mrcnn_integration"))

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

# %matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class FacesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "faces"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


config = FacesConfig()
config.display()

np.random.seed(0)
NUM_IMGS = 853  # images 0 through 852
IDS = np.arange(NUM_IMGS) + 1
np.random.shuffle(IDS)
TRAIN_IDS = IDS[:int(NUM_IMGS * .7)]  # [f'maksssksksss{x}' for x in IDS[:int(NUM_IMGS * .7)]]
VALID_IDS = IDS[int(NUM_IMGS * .7):int(NUM_IMGS * .9)]  #[f'maksssksksss{x}' for x in IDS[int(NUM_IMGS * .7):int(NUM_IMGS * .9)]]
TEST_IDS = IDS[int(NUM_IMGS * .9):]  #[f'maksssksksss{x}' for x in IDS[int(NUM_IMGS * .9):]]

TARGET_SHAPE = (128,128)
FILEPATH = './archive'


class FacesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required. <-- UPDATE THIS
    """

    def load_data(self, subset, filepath=FILEPATH):
        assert subset in ['train', 'val', 'test']
        self.add_class('masks', 0, 'no mask')
        self.add_class('masks', 1, 'mask')
        self.add_class('masks', 2, 'improper mask')

        # load ids of subsets
        if subset == 'train':
            image_ids = TRAIN_IDS
        elif subset == 'val':
            image_ids = VALID_IDS
        else:
            image_ids = TEST_IDS

        # add images
        for image_id in image_ids:
            self.add_image(source='masks', image_id=image_id, path=f'{FILEPATH}/images/{image_id - 1}.png')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info['source'] == 'masks':
            return info['masks']
        else:
            super(self.__class__).image_reference(self, image_id)
        return info

    def load_mask(self, image_id):
        def get_image_info(soup):
            info = self.image_info[image_id]
            info['height'] = int(soup.find('height').text)
            info['width'] = int(soup.find('width').text)
            info['boxes'] = soup.find_all('object')
            info['count'] = len(info['boxes'])
            return info

        f = open(f'{FILEPATH}/annotations/maksssksksss{image_id - 1}.xml')
        data = f.read()
        soup = BeautifulSoup(data, 'html.parser')
        info = get_image_info(soup)
        self.image_info[image_id]
        shapes = info['boxes']
        mask = np.zeros([info['height'], info['width'], info['count']], dtype=np.uint8)
        class_ids = []
        for i, (obj) in enumerate(info['boxes']):
            xmin = int(obj.find('xmin').text)
            ymin = int(obj.find('ymin').text)
            xmax = int(obj.find('xmax').text)
            ymax = int(obj.find('ymax').text)
            mask[ymin:ymax, xmin:xmax, i] = 1

            if obj.find('name').text == 'with_mask':
                class_ids.append(2)
            elif obj.find('name').text == 'without_mask':
                class_ids.append(1)
            else:
                class_ids.append(3)

        mask = np.array(mask)
        class_ids = np.array(class_ids)
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def load_image(self, image_id):
        img = Image.open(f'{FILEPATH}/images/maksssksksss{image_id - 1}.png')
        transform = torchvision.transforms.Resize(TARGET_SHAPE)
        img = np.asarray(transform(img))
        return img[:, :, :4] if img.shape[2] == 4 else img

# Training dataset
dataset_train = FacesDataset()
dataset_train.load_data(subset='train')
dataset_train.prepare()

# Validation dataset
dataset_valid = FacesDataset()
dataset_valid.load_data(subset='val')
dataset_valid.prepare()

# Testing dataset
dataset_test = FacesDataset()
dataset_test.load_data(subset='test')
dataset_test.prepare()

# # Create model in training mode
# model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
#
# # Which weights to start with?
# init_with = "imagenet"  #"coco"  # imagenet, coco, or last
#
# if init_with == "imagenet":
#     model.load_weights(model.get_imagenet_weights(), by_name=True)
# elif init_with == "coco":
#     # Load weights trained on MS COCO, but skip layers that
#     # are different due to the different number of classes
#     # See README for instructions to download the COCO weights
#     model.load_weights(COCO_MODEL_PATH) #, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
# elif init_with == "last":
#     # Load the last model you trained and continue training
#     model.load_weights(model.find_last(), by_name=True)
#
# # Train the head branches
# # Passing layers="heads" freezes all layers except the head
# # layers. You can also pass a regular expression to select
# # which layers to train by name pattern.
# model.train(dataset_train, dataset_valid,
#             learning_rate=config.LEARNING_RATE,
#             epochs=1,
#             layers='heads')
#
# # Fine tune all layers
# # Passing layers="all" trains all layers. You can also
# # pass a regular expression to select which layers to
# # train by name pattern.
# model.train(dataset_train, dataset_valid,
#             learning_rate=config.LEARNING_RATE / 10,
#             epochs=2,
#             layers="all")



class InferenceConfig(FacesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
image_id = random.choice(dataset_test.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_test, inference_config, image_id)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8), show_bbox=False)

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_test.class_names, r['scores'], ax=get_ax(), show_bbox=False)

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_test.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_test, inference_config, image_id)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))