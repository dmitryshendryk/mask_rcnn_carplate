import os 
import sys 
import numpy as np 
import tensorflow as tf 
import random 
import re 
import math 
import cv2
import matplotlib.pyplot as plt 

ROOT_DIR = os.path.abspath('../')

sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn.config import Config 
import mrcnn.model as modellib
from mrcnn import visualize


MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
COCO_MODEL_DIR = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')


class ShapeConfig(Config):

    NAME = 'shapes'
    CPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 3

    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    RPN_ANCHOR_SCALES = (8,16,32,64,128)

    TRAIN_ROIS_PER_IMAGE = 32 

    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 5 

config = ShapeConfig()
config.display()


def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


class ShapesDataset(utils.Dataset):

    def draw_shape_figures(self, img, shape, dims, color):
        x, y, s = dims

        if shape == 'squares':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
        elif shape == 'circle':
            cv2.circle(image, (x,y), s, color, -1)
        elif shape == 'traingle':
            points = np.array([[ 
                (x, y-s), 
                (x-s/math.sin(math.radians(60)), y+s),
                (x+s/math.sin(math.radians(60)), y+s),
            ]], dtype=np.int32) 

            cv2.fillPoly(image, points, color)



