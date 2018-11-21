import os 
import sys 
import numpy as np 
import tensorflow as tf 
import random 
import re 
import math 
import cv2
import matplotlib.pyplot as plt 


from utils import *
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize


config = ShapeConfig()
config.display()


def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax



if __name__ == "__main__":

    # Training dataset
    dataset_train = ShapesDataset()
    dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ShapesDataset()
    dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_val.prepare()

    # # Load and display random samples
    # image_ids = np.random.choice(dataset_train.image_ids, 4)
    # for image_id in image_ids:
    #     image = dataset_train.load_image(image_id)
    #     mask, class_ids = dataset_train.load_mask(image_id)
    #     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

    model.load_weights(COCO_MODEL_DIR, by_name=True,
                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])

    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=100, 
                layers='heads')