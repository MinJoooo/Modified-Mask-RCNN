"""
This code is modified 'Mask R-CNN' to suit my convenience.

Mask R-CNN :
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project and import MaskRCNN, COCO
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
import mrcnn.model as modellib
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco_new as coco

# Directory to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class DemoConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    BACKBONE = "resnet101"
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_DIM = 800
    MAX_GT_INSTANCES = 100
    MINI_MASK_SHAPE = (56, 56)
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    STEPS_PER_EPOCH = 1000
    TRAIN_ROIS_PER_IMAGE = 200


############################################################
#  Dataset
############################################################

def load_class_names():
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']
    return class_names


############################################################
#  Main function
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/dataset/",
                        help='Directory of the dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'last'")
    args = parser.parse_args()


    # Configurations and create model object in inference mode.
    config = DemoConfig()
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=DEFAULT_LOGS_DIR)


    # Select weights file to load
    if args.model.lower() == "last":
        model_path = model.find_last()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Load class_names
    class_names = load_class_names()
    print(class_names)

    # # Or you can load class_names from COCO dataset.
    # dataset = coco.CocoDataset()
    # dataset_path = "{}/..".format(args.dataset)
    # dataset.load_coco(dataset_path, "train")
    # dataset.prepare()
    # class_names = dataset.class_names
    # print(class_names)


    # Load a random image from the images folder
    file_name = random.choice(os.listdir(args.dataset))
    while "json" in file_name:
        file_name = random.choice(os.listdir(args.dataset))
    print("file name:", file_name)
    image = skimage.io.imread(os.path.join(args.dataset, file_name))


    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])


### It doesn't run if model is not proper or low accuracy.