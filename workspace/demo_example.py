import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import cv2

ROOT_DIR = os.path.abspath("../")

sys.path.append(ROOT_DIR)  
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/")) 
import coco


MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon_0029.h5")


IMAGE_DIR = os.path.join(ROOT_DIR, "balloon/val/")

print(COCO_MODEL_PATH)

class InfereceConfig(coco.Config):

    GPU_COUNT = 1
    IMAGE_PER_GPU = 1
    NAME = 'logs'
    NUM_CLASSES = 1 + 1

config = InfereceConfig()
config.display()



def start_detection():
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    model.load_weights(COCO_MODEL_PATH, by_name=True)


    class_names = ['BG', 'balloon']

    file_names = next(os.walk(IMAGE_DIR))[2]

    for file in file_names:
        image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))



        results = model.detect([image], verbose=1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'])
        
        cv2.waitKey(0)


start_detection()