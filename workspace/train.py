
"""
Mask R-CNN
Configurations and data loading code for MS COCO.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco
    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True
    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5
    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last
    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import json
import skimage
from PIL import ImageEnhance
#import font_recongisez
import cv2
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".


import zipfile
import urllib.request
import shutil
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from PIL import Image
from PIL import ImageEnhance
# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
# DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"

############################################################
#  Configurations
############################################################
# command='test'
command='train'
args = {'command':command,
            #'dataset':'/home/cui/桌面/gen_CAD_data/plc_fushion',
            # 'dataset':'/home/cui',
            'dataset': os.path.join(ROOT_DIR, "ccs_dataset_new"),
            'model':'mask_rcnn_coco.h5',
            'logs': os.path.join(ROOT_DIR, "logs"),
            'year':'2018'
            }

class PLCConfig(Config):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = "PLC"  # Override in sub-classes

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 100

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet101"

    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    # Number of classification classes (including background)
    NUM_CLASSES = 1 + 33  # Override in sub-classes

    # Length of square anchor side in pixels

    RPN_ANCHOR_SCALES = (32, 56, 72, 96, 128) #increase the anchor size

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.3,0.6,1]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.6

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 500

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = int(256)
    IMAGE_MAX_DIM = int(640)
    # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
    # up scaling. For example, if set to 2 then images are scaled up to double
    # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
    # Howver, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
    IMAGE_MIN_SCALE = 0

    # Image mean (RGB)
    MEAN_PIXEL = np.array([0.4491220455264695*225,0.4491220455264695*225, 0.4491220455264695*225])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 400

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 10

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped

    # this can be reduced to increase the possibality to be detected
    DETECTION_MIN_CONFIDENCE = 0.5

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.6

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.005
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 2,
        "mrcnn_class_loss": 2,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 0.5
    }

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    # Gradient norm clipping
    #归一化
    GRADIENT_CLIP_NORM = 5.0

from threading import Thread, Lock

class WebcamVideoStream :
    def __init__(self, src = 0, width = 1280, height = 720) :
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        #取消自动对焦
        self.stream.set(cv2.CAP_PROP_AUTOFOCUS,True)
        #self.stream.set(cv2.CAP_PROP_SETTINGS,1)
        #print(self.stream.set(cv2.CAP_PROP_FOCUS,5000))
        #focus = self.stream.get(cv2.CAP_PROP_FOCUS)
        # for i in range(10):
        #     print("focus length= %f"%focus)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

############################################################
#  Dataset--PLC
############################################################

class PLCDataset(utils.Dataset):

    def load_PLC(self, dataset_dir, subset):
        """Load a subset of the PLC dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have five class to add.
        self.add_class("PLC", 1, "")
        # self.add_class("PLC", 2, "B")
        # self.add_class("PLC", 3, "C")
        # self.add_class("PLC", 4, "D")
        # self.add_class("PLC", 5, "E")


        # Train or validation dataset?
        assert subset in ["json_train", "json_val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            # print('regions：',a['regions'].values)
            polygons = [a['regions'][r]['shape_attributes'] for r in range(len(a['regions']))]
            # polygons = [r['shape_attributes'] for r in a['regions'].values()]
            names = [a['regions'][r]['region_attributes']['name'] for r in range(len(a['regions']))]
            # 序列字典
            name_dict = {"A": 1}
            name_id = [name_dict[n] for n in names]


            # for index in range(len(polygons)):
            #     polygons[index]['name'] = a['regions'][index]['region_attributes']['name']
            # a['regions'][0]['region_attributes']['name'] = a['regions'][0]['region_attributes']['name']

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "PLC",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                class_id = name_id,
                polygons=polygons
                )

    def load_PLC_AutoLable(self, dataset_dir, subset):
        """Load a subset of the PLC dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have five class to add.
        #self.add_class("PLC", 1, "carplate")
        self.add_class("PLC",1, "0")
        self.add_class("PLC", 2, "1")
        self.add_class("PLC", 3, "2")
        self.add_class("PLC", 4, "3")
        self.add_class("PLC", 5, "4")
        self.add_class("PLC", 6, "5")
        self.add_class("PLC", 7, "6")
        self.add_class("PLC", 8, "7")
        self.add_class("PLC", 9, "8")
        self.add_class("PLC", 10, "9")
        self.add_class("PLC", 11, "A")
        self.add_class("PLC", 12, "B")
        self.add_class("PLC", 13, "C")
        self.add_class("PLC", 14, "D")
        self.add_class("PLC", 15, "E")
        self.add_class("PLC", 16, "F")
        self.add_class("PLC", 17, "G")
        self.add_class("PLC", 18, "H")
        #self.add_class("PLC", 19, "I")
        self.add_class("PLC", 20-1, "J")
        self.add_class("PLC", 21-1, "K")
        self.add_class("PLC", 22-1, "L")
        self.add_class("PLC", 23-1, "M")
        self.add_class("PLC", 24-1, "N")
        #self.add_class("PLC", 25, "O")
        self.add_class("PLC", 26-2, "P")
        #self.add_class("PLC", 27-2, "Q")
        self.add_class("PLC", 28-3, "R")
        self.add_class("PLC", 29-3, "S")
        self.add_class("PLC", 30-3, "T")
        self.add_class("PLC", 31-3, "U")
        self.add_class("PLC", 32-3, "V")
        self.add_class("PLC", 33-3, "W")
        self.add_class("PLC", 34-3, "X")
        self.add_class("PLC", 35-3, "Y")
        self.add_class("PLC", 36-3, "Z")

        #self.add_class("PLC", 37, "plate")

        # self.add_class("PLC", 3, "C")
        # self.add_class("PLC", 4, "D")
        # self.add_class("PLC", 5, "E")


        # Train or validation dataset?
        # assert subset in ["json_train2", "json_val2"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Add images
        print(dataset_dir)
        inputjson_f = open(os.path.join(dataset_dir, 'label.txt'), 'r')
        line = inputjson_f.readline()
        # if subset =="json_fusion-white_val":
        #     line = inputjson_f.readline()
        while line != "":
            tag=1
            lines_items = line.split(' ')
            img_name = lines_items[0]
            if not os.path.exists(dataset_dir+'/'+img_name):
                print("file doesn't exist:",subset,img_name)
                line = inputjson_f.readline()
                continue
            obj_num = lines_items[1]
            #print(obj_num)
            obj_start_index = 2
            obj_pts = 0
            polygons = []
            names = []
            for i in range(int(obj_num)):
                polygon = {}
                polygon['name'] = 'polygon'
                obj_name = lines_items[obj_start_index]
                # if len(img_name)==9 and int(img_name[5])>2:
                #     obj_name ="B"
                #     print("A--->B",img_name)
                print(obj_name)
                names.append(obj_name)
                obj_pts = int(lines_items[obj_start_index + 1])
                polygon['all_points_x'] = [int(lines_items[obj_start_index + 1 + j]) for j in range(1, obj_pts + 1)]
                polygon['all_points_y'] = [int(lines_items[obj_start_index + 1 + j]) for j in range(obj_pts + 1, obj_pts * 2 + 1)]
                for j in range(obj_pts + 1, obj_pts * 2 + 1):
                    if(int(lines_items[obj_start_index + 1 + j])>=720):
                        tag=0
                        break

                polygons.append(polygon)
                # print('img:', img_name, 'obj:', obj_name, 'obj pts:',
                #       [int(lines_items[obj_start_index + 1 + j]) for j in range(1, obj_pts * 2 + 1)])
                obj_start_index += obj_pts * 2 + 2
            line = inputjson_f.readline()
            if subset == "json_val_screen":
                line = inputjson_f.readline()
            # name_dict = {"A": 1, "B": 2}
            name_dict = {  "0": 1,
                         "1": 2,
                         "2": 3,
                         "3": 4,
                         "4": 5,
                         "5": 6,
                         "6": 7,
                         "7": 8,
                         "8": 9,
                         "9": 10,
                         "A": 11,
                         "B": 12,
                         "C": 13,
                         "D": 14,
                         "E": 15,
                         "F": 16,
                         "G": 17,
                         "H": 18,
                         #"I": 19,
                         "J": 20-1,
                         "K": 21-1,
                         "L": 22-1,
                         "M": 23-1,
                         "N": 24-1,
                         #"O": 25,
                         "P": 26-2,
                         #"Q": 27-2,
                         "R": 28-3,
                         "S": 29-3,
                         "T": 30-3,
                         "U": 31-3,
                         "V": 32-3,
                         "W": 33-3,
                         "X": 34-3,
                         "Y": 35-3,
                         "Z": 36-3,


                         }
            name_id = [name_dict[n] for n in names]


        # for a in annotations:
        #     # Get the x, y coordinaets of points of the polygons that make up
        #     # the outline of each object instance. There are stores in the
        #     # shape_attributes (see json format above)
        #     # print('regions：',a['regions'].values)
        #     polygons = [a['regions'][r]['shape_attributes'] for r in range(len(a['regions']))]
        #     # polygons = [r['shape_attributes'] for r in a['regions'].values()]
        #     names = [a['regions'][r]['region_attributes']['name'] for r in range(len(a['regions']))]
        #     # 序列字典
        #     name_dict = {"A": 1, "B": 2, "C": 3,"D":4,"E":5}
        #     name_id = [name_dict[n] for n in names]
        #
        #
        #     # for index in range(len(polygons)):
        #     #     polygons[index]['name'] = a['regions'][index]['region_attributes']['name']
        #     # a['regions'][0]['region_attributes']['name'] = a['regions'][0]['region_attributes']['name']
        #
        #     # load_mask() needs the image size to convert polygons to masks.
        #     # Unfortunately, VIA doesn't include it in JSON, so we must read
        #     # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, img_name)
            # image_path=image_path+'.jpg'
            if img_name=='screenshot-1534355667.png':
                s=0
            if not os.path.exists(image_path):
                continue
            image = skimage.io.imread(image_path)

            height, width = image.shape[:2]
            if tag==1:
                self.add_image(
                    "PLC",
                    image_id=img_name,  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    class_id = name_id,
                    polygons=polygons
                    )
                print('------------------------------------img:', img_name)


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "PLC":
            return super(self.__class__, self).load_mask(image_id)

        name_id = image_info["class_id"]
        print(name_id)
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_ids = np.array(name_id, dtype=np.int32)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            if 'all_points_y' in p.keys():
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            elif 'width' in p.keys():
                rr, cc = skimage.draw.polygon([p['y'],p['y'],p['y']+p['height'],p['height']],[p['x'],p['x']+p['width'],p['x']+p['width'],p['x']])
            mask[rr, cc, i] = 1

        # print( mask.astype(np.bool), name_id)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return (mask.astype(np.bool), class_ids)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "PLC":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = PLCDataset()
    # dataset_train.load_PLC_AutoLable(args['dataset'], "json_train2")
    # dataset_train.load_PLC_AutoLable(args['dataset'], "json_train_carplate")
    dataset_train.load_PLC_AutoLable(args['dataset'], "")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PLCDataset()
    # dataset_val.load_PLC(args['dataset'], "json_val")
    # dataset_val.load_PLC_AutoLable(args['dataset'], "json_train_carplate")
    dataset_val.load_PLC_AutoLable(args['dataset'], "")

    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1000,
                layers='all')

#
# ############################################################
# #  Dataset
# ############################################################
#
# class PLCDataset(utils.Dataset):
#     def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
#                   class_map=None, return_coco=False, auto_download=False):
#         """Load a subset of the COCO dataset.
#         dataset_dir: The root directory of the COCO dataset.
#         subset: What to load (train, val, minival, valminusminival)
#         year: What dataset year to load (2014, 2017) as a string, not an integer
#         class_ids: If provided, only loads images that have the given classes.
#         class_map: TODO: Not implemented yet. Supports maping classes from
#             different datasets to the same class ID.
#         return_coco: If True, returns the COCO object.
#         auto_download: Automatically download and unzip MS-COCO images and annotations
#         """
#
#         if auto_download is True:
#             self.auto_download(dataset_dir, subset, year)
#
#         coco = COCO("{}/PLC_data/annotations/instances_{}.json".format(dataset_dir, subset))
#         # coco = COCO(dataset_dir)
#         if subset == "minival" or subset == "valminusminival":
#             subset = "val"
#         image_dir = "{}/{}".format(dataset_dir, subset)
#
#         # Load all classes or a subset?
#         if not class_ids:
#             # All classes
#             class_ids = sorted(coco.getCatIds())
#
#         # All images or a subset?
#         if class_ids:
#             image_ids = []
#             for id in class_ids:
#                 image_ids.extend(list(coco.getImgIds(catIds=[id])))
#             # Remove duplicates
#             image_ids = list(set(image_ids))
#         else:
#             # All images
#             image_ids = list(coco.imgs.keys())
#
#         # Add classes
#         for i in class_ids:
#             self.add_class("coco", i, coco.loadCats(i)[0]["name"])
#
#         # Add images
#         for i in image_ids:
#             self.add_image(
#                 "coco", image_id=i,
#                 path=os.path.join(image_dir, coco.imgs[i]['file_name']),
#                 width=coco.imgs[i]["width"],
#                 height=coco.imgs[i]["height"],
#                 annotations=coco.loadAnns(coco.getAnnIds(
#                     imgIds=[i], catIds=class_ids, iscrowd=None)))
#         if return_coco:
#             return coco
#
#     def auto_download(self, dataDir, dataType, dataYear):
#         """Download the COCO dataset/annotations if requested.
#         dataDir: The root directory of the COCO dataset.
#         dataType: What to load (train, val, minival, valminusminival)
#         dataYear: What dataset year to load (2014, 2017) as a string, not an integer
#         Note:
#             For 2014, use "train", "val", "minival", or "valminusminival"
#             For 2017, only "train" and "val" annotations are available
#         """
#
#         # Setup paths and file names
#         if dataType == "minival" or dataType == "valminusminival":
#             imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
#             imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
#             imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
#         else:
#             imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
#             imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
#             imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
#         # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)
#
#         # Create main folder if it doesn't exist yet
#         if not os.path.exists(dataDir):
#             os.makedirs(dataDir)
#
#         # Download images if not available locally
#         if not os.path.exists(imgDir):
#             os.makedirs(imgDir)
#             print("Downloading images to " + imgZipFile + " ...")
#             with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
#                 shutil.copyfileobj(resp, out)
#             print("... done downloading.")
#             print("Unzipping " + imgZipFile)
#             with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
#                 zip_ref.extractall(dataDir)
#             print("... done unzipping")
#         print("Will use images in " + imgDir)
#
#         # Setup annotations data paths
#         annDir = "{}/annotations".format(dataDir)
#         if dataType == "minival":
#             annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
#             annFile = "{}/instances_minival2014.json".format(annDir)
#             annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
#             unZipDir = annDir
#         elif dataType == "valminusminival":
#             annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
#             annFile = "{}/instances_valminusminival2014.json".format(annDir)
#             annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
#             unZipDir = annDir
#         else:
#             annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
#             annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
#             annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
#             unZipDir = dataDir
#         # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)
#
#         # Download annotations if not available locally
#         if not os.path.exists(annDir):
#             os.makedirs(annDir)
#         if not os.path.exists(annFile):
#             if not os.path.exists(annZipFile):
#                 print("Downloading zipped annotations to " + annZipFile + " ...")
#                 with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
#                     shutil.copyfileobj(resp, out)
#                 print("... done downloading.")
#             print("Unzipping " + annZipFile)
#             with zipfile.ZipFile(annZipFile, "r") as zip_ref:
#                 zip_ref.extractall(unZipDir)
#             print("... done unzipping")
#         print("Will use annotations in " + annFile)
#
#     def load_mask(self, image_id):
#         """Load instance masks for the given image.
#
#         Different datasets use different ways to store masks. This
#         function converts the different mask format to one format
#         in the form of a bitmap [height, width, instances].
#
#         Returns:
#         masks: A bool array of shape [height, width, instance count] with
#             one mask per instance.
#         class_ids: a 1D array of class IDs of the instance masks.
#         """
#         # If not a COCO image, delegate to parent class.
#         image_info = self.image_info[image_id]
#         if image_info["source"] != "coco":
#             return super(CocoDataset, self).load_mask(image_id)
#
#         instance_masks = []
#         class_ids = []
#         annotations = self.image_info[image_id]["annotations"]
#         # Build mask of shape [height, width, instance_count] and list
#         # of class IDs that correspond to each channel of the mask.
#         for annotation in annotations:
#             class_id = self.map_source_class_id(
#                 "coco.{}".format(annotation['category_id']))
#             if class_id:
#                 m = self.annToMask(annotation, image_info["height"],
#                                    image_info["width"])
#                 # Some objects are so small that they're less than 1 pixel area
#                 # and end up rounded out. Skip those objects.
#                 if m.max() < 1:
#                     continue
#                 # Is it a crowd? If so, use a negative class ID.
#                 if annotation['iscrowd']:
#                     # Use negative class ID for crowds
#                     class_id *= -1
#                     # For crowd masks, annToMask() sometimes returns a mask
#                     # smaller than the given dimensions. If so, resize it.
#                     if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
#                         m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
#                 instance_masks.append(m)
#                 class_ids.append(class_id)
#
#         # Pack instance masks into an array
#         if class_ids:
#             mask = np.stack(instance_masks, axis=2).astype(np.bool)
#             class_ids = np.array(class_ids, dtype=np.int32)
#             return mask, class_ids
#         else:
#             # Call super class to return an empty mask
#             return super(CocoDataset, self).load_mask(image_id)
#
#     def image_reference(self, image_id):
#         """Return a link to the image in the COCO Website."""
#         info = self.image_info[image_id]
#         if info["source"] == "coco":
#             return "http://cocodataset.org/#explore?id={}".format(info["id"])
#         else:
#             super(CocoDataset, self).image_reference(image_id)
#
#     # The following two functions are from pycocotools with a few changes.
#
#     def annToRLE(self, ann, height, width):
#         """
#         Convert annotation which can be polygons, uncompressed RLE to RLE.
#         :return: binary mask (numpy 2D array)
#         """
#         segm = ann['segmentation']
#         if isinstance(segm, list):
#             # polygon -- a single object might consist of multiple parts
#             # we merge all parts into one mask rle code
#             rles = maskUtils.frPyObjects(segm, height, width)
#             rle = maskUtils.merge(rles)
#         elif isinstance(segm['counts'], list):
#             # uncompressed RLE
#             rle = maskUtils.frPyObjects(segm, height, width)
#         else:
#             # rle
#             rle = ann['segmentation']
#         return rle
#
#     def annToMask(self, ann, height, width):
#         """
#         Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
#         :return: binary mask (numpy 2D array)
#         """
#         rle = self.annToRLE(ann, height, width)
#         m = maskUtils.decode(rle)
#         return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results
pannel_index=0
total_count=1
pcl_container=[]
import math
#注意 这里采用了非极大值抑制  后续取消
all_labels=["SM 1234","SM 1231","SM 1223"]
def display_mask_image(cv_window_name,image, boxes, masks, class_ids, class_names,
                      scores=None, title="PLC",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None,
                      score_threshold=0.8,show_score=True,Traceing_threshold=150):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    print(N)
    global pcl_container
    global  total_count
    if not N:
        print("\n*** No instances to display *** \n")
        return image
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    max_score=0
    index=0
    total_height=0
    for i in range(N):
        if scores[i]>max_score and class_names[class_ids[i]]!="carplate":
           y1, x1, y2, x2 = boxes[i]
           height=y2-y1
           total_height+=height
    average_height=total_height/N

    # If no axis is passed, create one and automatically call show()
    auto_show = True
    # if not ax:
    #     _, ax = plt.subplots(1, figsize=figsize)
    #     auto_show = True
    #
    # Generate random colors
    # colors = visualize.fixed_colors(N)

    colors = visualize.get_colors(38)
    print(colors)
    # Show area outside image boundaries.
    # height, width = image.shape[:2]
    # ax.set_ylim(height + 10, -10)
    # ax.set_xlim(-10, width + 10)
    # ax.axis('off')
    # ax.set_title(title)
    scoreMin=score_threshold
    masked_image=image.astype(np.uint8).copy()
    #cv2.imshow(" ",masked_image)
    #cv2.waitKey(0)
    for i in range(N):
        #i=index
        if scores[index] > scoreMin:
            # Mask
            #print(class_names[class_ids[i]])
            mask = masks[:, :, i]
            #color = colors[class_ids[i]]
            color =[1,1,1]
            # color =
            if show_mask:
                masked_image = visualize.apply_mask(image, mask, color,alpha=0.4)

    masked_image = masked_image.astype(np.uint8).copy()
    ls=[]
    bs=[]
    ss=[]
    max_car_plate_score = -1
    car_plate_pos = []
    for i in range(N):
        #i = index
        #print(i)
        if scores[i]>scoreMin:
            print(class_names[class_ids[i]]+' scores:', scores[i])
            # color = colors[i]
            #color = colors[class_ids[i]]
            color=(1,1,1)
            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            class_id = class_ids[i]
            sc = scores[i]
            #print(class_names)
            label=class_names[class_id]
            box=[(x1,y1),(x2,y2)]


            plate_max_score=-1
            #car_plate_score=[]

            if label!='carplate':
                bs.append(box)
                ls.append(label)
                ss.append(sc)
            elif sc>max_car_plate_score:
                car_plate_pos=[y1, x1, y2, x2]
                plate_max_score=sc



            #la='SIMATIC S7-1200'
            #sc='score :: %.3f'%sc


            if show_bbox :
               # if len(car_plate_pos):
               #  masked_image = cv2.rectangle(masked_image, (car_plate_pos[1], car_plate_pos[0]),(car_plate_pos[3], car_plate_pos[2]),(100,20,100),thickness=2)












                #center=[(x2-x1)/2+x1,(y2-y1)/2+y1]

                minDis=9999
                minindex=-1
                # for plc in pcl_container:
                #     dist=math.sqrt(math.pow(center[0]-plc.center[0],2)+math.pow(center[1]-plc.center[1],2))
                #     print("dist   %d"%dist)
                #     if dist/plc.diag>0.5:
                #         continue
                #     elif dist<minDis:
                #         minDis=dist/plc.diag
                #         minindex=pcl_container.index(plc)
                # mark=-1
                # diag = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
                # if minindex==-1:
                #
                #     pcl_container.append(font_recongisez.PLC(center,index=total_count,diag=diag))
                #     mark=len(pcl_container)-1
                #     total_count+=1
                # else:
                #     pcl_container[minindex].set_center(center)
                #     pcl_container[minindex].diag=diag
                #     pcl_container[minindex].set_need_update(False)
                #     mark=minindex
                #front_pannel=masked_image[y1:y2,x1:x2]
                # cv2.imwrite('/media/cui/Ubuntu/front_pannel_snap/%d.png'%pannel_index,front_pannel)

                #cv2.imshow("pannel",front_pannel)


                # global all_labels
                # cv2.putText(masked_image, la, (x1, y1 - 25), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
                # if not pcl_container[mark].get_sub():
                #  name = font_recongisez.recongnise(image=front_pannel)
                #  if name in all_labels:
                #      pcl_container[mark].set_sub(name)
                #     cv2.putText(masked_image, name, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
                #     global pannel_index
                #     cv2.imwrite("/media/cui/Ubuntu/demo/%d.png"%pannel_index,masked_image)
                #     global pannel_index
                #     pannel_index += 1
                #cv2.putText(masked_image, pcl_container[mark].get_sub(), (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
                # p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                #                     alpha=0.7, linestyle="dashed",
                #                     edgecolor=color, facecolor='none')
                # ax.add_patch(p)

            # Label
            if not captions:
                class_id = class_ids[i]
                score = scores[index] if scores is not None else None
                #print(class_names)
                label = class_names[class_id]
                # x = random.randint(x1, (x1 + x2) // 2)
                #caption = "{}  {:.3f}".format(label, score) if score else label
                caption = "{}".format(label)
            else:
                caption = captions[i]

            #image = cv2.addText(masked_image,caption, (x1+10, y1+10),'Times',pointSize=20,color=color)
            # ax.text(x1, y1 + 8, caption,
            #         color='w', size=11, backgroundcolor="none")


            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = visualize.find_contours(padded_mask, 0.5)
            # for verts in contours:
            #     # Subtract the padding and flip (y, x) to (x, y)
            #     verts = np.fliplr(verts) - 1
            #     print('verts:',verts)
                # points = np.array([[(1,1),(20,20),(15,15)]], dtype=np.int32)
                # masked_image = cv2.fillPoly(masked_image,points,color)
                # p = Polygon(verts, facecolor="none", edgecolor=color)
                # ax.add_patch(p)
        # ax.imshow(masked_image.astype(np.uint8))
        # cv2.imshow(cv_window_name, masked_image)
        # cv2.waitKey(1)
        # if auto_show:
        #     plt.show()
    # for i in pcl_container:
    #     if i.is_need_update():
    #         if i.ls_frame==10:
    #          pcl_container.remove(i)
    #         else:
    #          i.ls_frame+=1
    #     else:
    #         i.set_need_update(True)
            #cv2.putText(masked_image, "No %d"%i.index,(int(i.center[0]),int(i.center[1])), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

    #total = "now have %d PLC in scene" % len(pcl_container)
    #cv2.putText(masked_image, total, (0, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
    v_t=average_height*0.6
    h_t=average_height
    if car_plate_pos:
     masked_image = cv2.rectangle(masked_image, (car_plate_pos[1], car_plate_pos[0]),
                                 (car_plate_pos[3], car_plate_pos[2]), (100, 20, 100), thickness=2)
    res=sequence(ls,bs,ss,v_t,h_t,image.shape)
    first=""
    sec=""
    if len(res)>1:
        for c in res[0]:
            first+=c
        for d in res[1]:
            sec+=d
        if len(first)>len(sec):
            temp=first
            first=sec
            sec=temp
    elif len(res)==1:
        if res[0]!="":
            for d in res[0]:
                sec += d
    print(first,sec)
    #cv2.imshow(" ",masked_image)
    #cv2.waitKey(0)
    starty=20
    if first!="":
        cv2.putText(masked_image,first,(0,starty),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,255))
        starty+=20
    cv2.putText(masked_image, sec, (0,starty), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255))
    return masked_image

from mrcnn.model import log


def debug_model(model,config,image,gt_class_id,gt_bbox):
    target_rpn_match, target_rpn_bbox = modellib.build_rpn_targets(
        image.shape, model.anchors, gt_class_id, gt_bbox, model.config)
    log("target_rpn_match", target_rpn_match)
    log("target_rpn_bbox", target_rpn_bbox)

    positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
    negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
    neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
    positive_anchors = model.anchors[positive_anchor_ix]
    negative_anchors = model.anchors[negative_anchor_ix]
    neutral_anchors = model.anchors[neutral_anchor_ix]
    log("positive_anchors", positive_anchors)
    log("negative_anchors", negative_anchors)
    log("neutral anchors", neutral_anchors)

    # Apply refinement deltas to positive anchors
    refined_anchors = utils.apply_box_deltas(
        positive_anchors,
        target_rpn_bbox[:positive_anchors.shape[0]] * model.config.RPN_BBOX_STD_DEV)
    log("refined_anchors", refined_anchors, )
    visualize.draw_boxes(image, boxes=positive_anchors, refined_boxes=refined_anchors, ax=get_ax())

def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)



def preprocess_image(source_image,width,height):
    resized_image = cv2.resize(source_image, (int(width/4), int(height/4)))
    # trasnform values from range 0-255 to range -1.0 - 1.0
    # resized_image = resized_image - 127.5
    # resized_image = resized_image * 0.007843
    return resized_image
def space_NMS(box_a,box_b):#((x1,y1),(x2,y2))
    width_a=abs(box_a[0][0]-box_a[1][0])
    width_b=abs(box_b[0][0]-box_b[1][0])
    height_a=abs(box_a[0][1]-box_a[1][1])
    height_b=abs(box_b[0][1]-box_b[1][1])
    size_a=width_a*height_a
    size_b=width_b*height_b
    start_x=max(box_a[0][0],box_b[0][0])
    end_x=min(box_a[1][0],box_b[1][0])
    start_y = max(box_a[0][1], box_b[0][1])
    end_y= min(box_a[1][1], box_b[1][1])

    #size_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    center_a=((box_a[0][0]+box_a[1][0])/2,(box_a[0][1]+box_a[1][1])/2)
    center_b=((box_b[0][0]+box_b[1][0])/2,(box_b[0][1]+box_b[1][1])/2)
    if start_x>end_x or start_y>end_y:
        #no overlap
        #print(center_a,center_b)
        return False
    else:

        # overlapsize=((width_a+width_b)-2*(abs(center_a[0]-center_b[0])))*((height_a+height_b)-2*(abs(center_a[1]-center_b[1])))
        # overlapsize=(0.5*(width_a+width_b)-(center_b[0]-center_a[0]))*(0.5*(height_a+height_b)-(center_b[1]-center_a[1]))
        overlapsize=abs(end_x-start_x)*abs(end_y-start_y)
        print("overlapsize: ", overlapsize, " size_b: ", size_b)
        if overlapsize>=0.7*size_b or overlapsize>=0.7 *size_a:

            return  True
        else:
            return False

def aggregate(line,labels,scores,boxs,h_thershold):
    opt_label=[]
    temps=[]
    print(line,labels,scores,boxs)
    while(len(line)):
        mark = []
        pos=line[0][0]
        label=labels[0]
        score=scores[0]
        box=boxs[0]
        #mark.append(0)

        for i in range(1,len(line),1):
            if not space_NMS(box,boxs[i]):
                mark.append(i)
            elif scores[i]>score:
                    print("label: ", label)
                    label=labels[i]
                    score=scores[i]
            else:
                print("label: ",labels[i])
                continue
        newline=[]
        newlabels=[]
        newscores=[]
        newbox=[]
        #print(mark)
        for i in mark:
            newline.append(line[i])
            newlabels.append(labels[i])
            newscores.append(scores[i])
            newbox.append(boxs[i])
        line=newline
        labels=newlabels
        scores=newscores
        boxs=newbox
        temps.append((pos,label))
        #mark.clear()
    temps.sort(key=lambda tu:tu[0])
    for t in temps:
        opt_label.append(t[1])
    return opt_label
##this function is aim to sequence the input result
import skimage.transform as st
import  math

#this is a backup
def Seperate_V(centers, imgsize, boxs, scores, labels):
    output_lines = []
    output_labels = []
    output_boxs = []
    output_scores = []
    if (len(centers) < 2):
        return output_lines, output_labels, output_scores, output_boxs
    point_img = np.zeros((imgsize[0], imgsize[1]))

    for center in centers:
        point_img[int(center[1]), int(center[0])] = 255
    cv2.imshow(" ",point_img)
    cv2.waitKey(0)
    h, theta, d = st.hough_line(point_img)
    k = -1
    b = []

    # in same theata the difference of d should less than the thersehold
    first_line = []
    second_line = []
    average = 9999
    for j in range(h.shape[1]):  # d
        all_dis = h[:, j]

        previous = -1
        alldis = []

        for i in range(len(all_dis)):
            apperance = all_dis[i]
            while (apperance):
                alldis.append(d[i])
                apperance -= 1
        th = 5 # 不允许超过0.1
        count = 0
        #print("alldis",alldis)
        temp_d = [alldis[0]]
        sum = 0
        for i in range(1, len(alldis), 1):
            sum += abs(alldis[i] - alldis[i - 1])
            if abs(alldis[i] - alldis[i - 1]) > th:
                temp_d.append(alldis[i])
                count += 1
        temp_average = sum / len(alldis)
        if count <= 1 and temp_average < average:
            k = theta[j]
            b = temp_d
            average = temp_average
        # if count<=1:
        #     #print(j,temp_d)
        #     k=j
        #     b=temp_d
        #     break

    print(k,b)
    if not len(b):
        return output_lines, output_labels, output_scores, output_boxs
    if len(b) == 1:
        output_lines = [centers]
        output_boxs = [boxs]
        output_labels = [labels]
        output_scores = [scores]
    else:
        if k == 0:
            k = 1
        cos = math.cos(k)
        sin = math.sin(k)
        output_lines = [[], []]
        output_labels = [[], []]
        output_boxs = [[], []]
        output_scores = [[], []]
        for i in range(len(centers)):
            # print(cos/sin*i[0]+b[0]/sin,cos/sin*i[0]+b[1]/sin)
            if abs(centers[i][1] + cos / sin * centers[i][0] - b[0] / sin) > abs(
                    centers[i][1] + cos / sin * centers[i][0] - b[1] / sin):
                output_lines[0].append(centers[i])
                output_labels[0].append(labels[i])
                output_boxs[0].append(boxs[i])
                output_scores[0].append(scores[i])
            else:
                output_lines[1].append(centers[i])
                output_labels[1].append(labels[i])
                output_boxs[1].append(boxs[i])
                output_scores[1].append(scores[i])
    return output_lines, output_labels, output_scores, output_boxs

def sequence(labels, boxs, scores, v_thershold, h_thershold, size=0):
    # first determine wether the car plate is two lines
    is_two_lines = False

    centers = []
    for box in boxs:
        center = [(box[0][0] + box[1][0]) / 2.0, (box[0][1] + box[1][1]) / 2.0]
        centers.append(center)
    # check y
    la = []
    sc = []
    lines = []
    all_boxes = []
    output = []
    # print(centers,labels,scores)

    lines, la, sc, all_boxes = Seperate_V(centers, size, boxs, scores, labels)
  
    newline = []
    newscores = []
    newlabels = []
    newboxs = []
    for i in range(len(lines)):
        line = lines[i]
        score = sc[i]
        label = la[i]
        c_box = all_boxes[i]
        # print(c_box)
        if len(line) >= 2:  # at least 2
            newline.append(line)
            newscores.append(score)
            newlabels.append(label)
            newboxs.append(c_box)
    # determine x

    for i in range(len(newline)):
        line = newline[i]
        label_line = newlabels[i]
        score_line = newscores[i]
        box_line = newboxs[i]
        code = aggregate(line, label_line, score_line, box_line, h_thershold)
        output.append(code)

    if len(output) > 2:
        # print(output)
        output = []

    return output


count=1
def detect(model, image_path=None, video_path=None,image_dir = None,camera=None,Min_score = 0.8):
    assert image_path or video_path or image_dir or camera
    if image_dir:
        # write_img=1
        write_img = 0
        imgs = os.listdir(image_dir)
        for img in imgs:
            if img[-1]=='t':
                continue
            # Run model detection and generate the color splash effect
            print("Running on {}".format(img))
            # Read image
            # image = skimage.io.imread(image_dir + '/' + img)
            # image2 = skimage.io.imread(image_dir + '/' + img,as_grey=True)
            image = cv2.imread(image_dir + '/' + img,cv2.IMREAD_GRAYSCALE)
            # for i in range(5):
            #     average=int((np.max(image)+np.min(image))*0.5)
            #     print(average)
            #     for r in range(image.shape[0]):
            #         for c in range(image.shape[1]):
            #             if image[r][c] > average:
            #                 temp = 200
            #                 # if temp > 255:
            #                 #     temp = 255
            #             else:
            #                 continue
            #             image[r][c] = temp
            s=(2*image.shape[1],2*image.shape[0])
            image=cv2.resize(src=image,dsize=s,interpolation=cv2.INTER_LANCZOS4)
            #image = cv2.bilateralFilter(src=image, d=0, sigmaColor=10, sigmaSpace=7)
            #image=cv2.GaussianBlur(image,(7,7),0)
            # for i in range(1):
            #     average=int((np.max(image)+np.min(image))*0.5)
            #     print(average)
            #     for r in range(image.shape[0]):
            #         for c in range(image.shape[1]):
            #             if image[r][c] > average:
            #                 temp = 200
            #                 # if temp > 255:
            #                 #     temp = 255
            #             else:
            #                 continue
            #             image[r][c] = temp

            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
            # kernel = np.ones((5, 5), np.uint8)
            # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)



            # enh_con = ImageEnhance.Contrast(image)
            # contrast = 1.5
            # image_contrasted = enh_con.enhance(contrast)
            # image=image_contrasted



            # cv2.imshow(" ",image)
            # cv2.waitKey(0)
            # image =  skimage.io.imread(image_dir+'/'+img)
            # OpenCV returns images as BGR, convert to RGB
            image = image[..., ::-1]
            # image = image[..., ::-1]
            # image = image[..., ::- 1]
            # Detect objects
            t1 = time.time()
            r = model.detect([image], verbose=1)[0]
            print('detect time: ', time.time() - t1)
            cv_window_name = "Mask-RCNN for PLC"
            cv2.namedWindow(cv_window_name)
            # cv2.moveWindow(cv_window_name, 10, 10)
            # class_names = ['BG','0', '1','2','3', '4','5','6', '7','8','9', 'A','B','C', 'D','E','F', 'G','H','I', 'J','K','L', 'M','N','O', 'P','Q','R', 'S','T','U', 'V','W','X', 'Y','Z',
            #                ]
            class_names = [ "BG",
                           "0",
                           "1",
                           "2",
                           "3",
                           "4",
                           "5",
                           "6",
                           "7",
                           "8",
                           "9",
                           "A",
                           "B",
                           "C",
                           "D",
                           "E",
                           "F",
                           "G",
                           "H",
                           "J",
                           "K",
                           "L",
                           "M",
                           "N",
                           "P",
                           "R",
                           "S",
                           "T",
                           "U",
                           "V",
                           "W",
                           "X",
                           "Y",
                           "Z",


                           ]
                           #                ]
            # print('redult:',list(r))
            mask_img = display_mask_image(cv_window_name, image, r['rois'], r['masks'], r['class_ids'],
                                          class_names, r['scores'], show_bbox=True,score_threshold=Min_score,show_mask=False)
            # print('PLC_data/masked_p1/masked_'+img[:-4])
            write_img=1
            if write_img==1:
                global count
                cv2.imwrite("/home/jianfenghuang/Desktop/VAL_LOG/bench_mark_1119/{}.png".format(count),mask_img)
                count += 1
            # foutput.writelines(img + ':\n')
            str = r['rois']
            # np.savetxt('PLC_data/masked_txt/'+img[:-4]+'.txt',str, fmt='%.2f')

            # pic = visualize.m_apply_mask(image,r['masks'])
            # if write_img == 1:
            #     arr = np.zeros([720, 1280])
            #
            #     for index in range(r['masks'].shape[2]):
            #         if r['scores'][index] > 0.8:
            #             mask = r['masks'][:, :, index].reshape([720, 1280])
            #             arr[mask] = 255
            #     cv2.imwrite('PLC_data/bin_map/'+img[:-4]+'.jpg',arr)
            # np.savetxt('test.txt', a, fmt='%.2f')
            # foutput.writelines('\n')
            cv2.imshow(cv_window_name, mask_img)

            #cv2.imwrite("/media/cui/Ubuntu/ccs_data_set/true_test_characters/{}.png".format(str(count)),mask_img)

            cv2.waitKey(0)
        # foutput.close()
    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # # OpenCV returns images as BGR, convert to RGB
        # image = image[..., ::-1]
        # Detect objects
        t1 = time.time()
        r = model.detect([image], verbose=1)[0]
        print('detect time: ', time.time() - t1)
        cv_window_name = "Mask-RCNN for PLC"
        cv2.namedWindow(cv_window_name)
        # cv2.moveWindow(cv_window_name, 10, 10)
        class_names   = [ "bcakground",
                           "carplate",
                           "1",
                           "2",
                           "3",
                           "4",
                           "5",
                           "6",
                           "7",
                           "8",
                           "9",
                           "A",
                           "B",
                           "C",
                           "D",
                           "E",
                           "F",
                           "G",
                           "H",
                           "I"
                           "J",
                           "K",
                           "L",
                           "M",
                           "N",
                           "O"
                           "P",
                           "Q",
                           "R",
                           "S",
                           "T",
                           "U",
                           "V",
                           "W",
                           "X",
                           "Y",
                           "Z",
                           "0"

                           ]
        print("0000000000000000000000000000000000")
        mask_img = display_mask_image(cv_window_name, image, r['rois'], r['masks'], r['class_ids'],
                                      class_names, r['scores'],show_bbox=True,score_threshold=Min_score,show_mask=False)

        cv2.imshow(cv_window_name, mask_img)
        cv2.waitKey(0)
        # Color splash
        # splash = color_splash(image, r['masks'])
        # # Save output
        # file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        # skimage.io.imsave(file_name, splash)
    elif video_path:

        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        # file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        # vwriter = cv2.VideoWriter(file_name,
        #                           cv2.VideoWriter_fourcc(*'MJPG'),
        #                           fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()

            if success:
                image = preprocess_image(image, width, height)
                # OpenCV returns images as BGR, convert to RGB
                # image = image[..., ::-1]
                # Detect objects
                t1 = time.time()
                r = model.detect([image], verbose=0)[0]
                print('detect time: ', time.time() - t1)
                cv_window_name = "Mask-RCNN for PLC"
                cv2.namedWindow(cv_window_name)
                # cv2.moveWindow(cv_window_name, 10, 10)
                class_names = ['BG', 'Simatic IOT2000', 'Raspberry Pi', 'Nuc', 'DR-120-24', 'RS30']
                # print('redult:',list(r))
                mask_img = display_mask_image(cv_window_name, image, r['rois'], r['masks'], r['class_ids'],
                                                       class_names, r['scores'],score_threshold=0.95)

                cv2.imshow(cv_window_name, mask_img)
                cv2.waitKey(1)
                count += 1
            else:
                print("video stop!")
                break
        # vwriter.release()
        cv2.destroyAllWindows()
    if camera:
        count = 1
        while (True):
            # try:
                image = camera.read()
                #con=ImageEnhance.Contrast(image)
                #contrast=1.5
                #image1=con.enhance(contrast)

                # OpenCV returns images as BGR, convert to RGB-
                # image = image[..., ::-1]
                #image = preprocess_image(image, 720, 1280)
                t1 = time.time()
                r = model.detect([image], verbose=0)[0]
                print('detect time: ', time.time() - t1)
                cv_window_name = "Mask-RCNN for PLC"
                cv2.namedWindow(cv_window_name)
                # cv2.moveWindow(cv_window_name, 10, 10)
                class_names  = [ "bcakground",
                           "carplate",
                           "1",
                           "2",
                           "3",
                           "4",
                           "5",
                           "6",
                           "7",
                           "8",
                           "9",
                           "A",
                           "B",
                           "C",
                           "D",
                           "E",
                           "F",
                           "G",
                           "H",
                           "I"
                           "J",
                           "K",
                           "L",
                           "M",
                           "N",
                           "O"
                           "P",
                           "Q",
                           "R",
                           "S",
                           "T",
                           "U",
                           "V",
                           "W",
                           "X",
                           "Y",
                           "Z",
                           "0"

                           ]
                # print('redult:',list(r))
                mask_img = display_mask_image(cv_window_name, image, r['rois'], r['masks'], r['class_ids'],
                                              class_names, r['scores'],score_threshold=Min_score,show_mask=False)

                cv2.imshow(cv_window_name, mask_img)
                cv2.waitKey(1)
                count += 1
            # except:
            #     print("video stop!")
            #    break
        cv2.destroyAllWindows()

    # print("Saved to ", file_name)
############################################################
#  Training
############################################################


if __name__ == '__main__':

 
    config = PLCConfig()

    # config.display()

    # Create model
    if args['command'] == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=os.path.join(ROOT_DIR, 'logs'))
    else:

        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args['logs'])

    # # Select weights file to load
    # if args['model'].lower() == "coco":
    #     model_path = COCO_MODEL_PATH
    # elif args['model'].lower() == "last":
    #     # Find last trained weights
    #     model_path = model.find_last()
    # elif args['model'].lower() == "imagenet":
    #     # Start from ImageNet trained weights
    #     model_path = model.get_imagenet_weights()
    # else:
    #     model_path = args['model']

    # train or test
    if args['command'] == "train":
        # Load weights
        #print("Loading weights ", model_path)

        # model_path = '/home/cui/桌面/PythonEnvironment/Mask_rcnn_project/logs/plc20180824T1957/mask_rcnn_plc_0062.h5'
        # model.load_weights(model_path, by_name=True)
        # Exclude the last layers because they require a matching
        # number of classes
        # model_path = "char_101.h5"
        #model.load_weights(model_path, by_name=True)
        # model.load_weights(model_path, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
        # "mrcnn_bbox", "mrcnn_mask"])#这几个一定要exclude掉 全连接层的维度不对 当前尝试训练所有的  其实对resnet来说 只需要训练高层的就好吧
        # exclude = [
        #     "mrcnn_class_logits", "mrcnn_bbox_fc",
        #     "mrcnn_bbox", "mrcnn_mask"]

        train(model)
    elif args['command'] == "test":

        # test_model_path = '/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/logs/plc20180808T1324/mask_rcnn_plc_0017.h5'
        #test_model_path = '/home/jianfenghuang/Desktop/VAL_LOG/mask_rcnn_plc_0184.h5'

        #test_model_path = '/home/jianfenghuang/Desktop/VAL_LOG/PCL_LOGS/plc20181127T1104/mask_rcnn_plc_0095.h5'
        #test_model_path="/homeengh/jianfuang/Desktop/weights/this_is_the_best_char_weight.h5"
        test_model_path="/home/jianfenghuang/Desktop/VAL_LOG/PCL_LOGS/plc20181219T1758/mask_rcnn_plc_0300.h5"
        test_model_path="/home/jianfenghuang/Desktop/VAL_LOG/PCL_LOGS/plc20181220T1723/mask_rcnn_plc_0644.h5"
        print("Loading weights ", test_model_path)
        model.load_weights(test_model_path, by_name=True)
        #imgPath = '/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/PLC_data/colorimage_1.jpg'
        #image_dir='/media/cui/Ubuntu/ccs_data_set/Bigscale_train'
        #image_dir = '/home/cui/桌面/GenerateImgs/Bigscale_train'
        #image_dir = '/home/jianfenghuang/Desktop/ccs_dataset'
        #image_dir='/home/jianfenghuang/Pictures/car2'
        image_dir='/home/jianfenghuang/Desktop/VAL_LOG/true_test_characters'
        #image_dir='/home/jianfenghuang/Desktop/car_plate_pictures'
        #image_dir='/home/jianfenghuang/Pictures/ecb_test'
        #image_dir = '/home/jianfenghuang/Downloads/crop'
        # image_dir = '/home/cui/图片/'
        #image_dir='/home/cui/下载/crop'
        #image_dir = '/home/cui/桌面/pcl'
        front_surface='/home/cui/front_only/front_pannel/'
        detect(model, image_dir=image_dir,Min_score=0.7 )
        print('detecting...')
        #detect(model,video_path='/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/PLC_data/VID_20180730_095913.mp4')


        # # from camera
        vs = WebcamVideoStream(src=0).start()

        detect(model, camera=vs,Min_score=0.95)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
