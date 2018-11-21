import os 
import sys 

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)

from mrcnn.config import Config 



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