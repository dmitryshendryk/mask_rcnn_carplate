import os 
import sys 
import numpy as np 
import random
import cv2

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)

from mrcnn.config import Config 
from mrcnn import utils



MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
COCO_MODEL_DIR = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')


class ShapesDataset(utils.Dataset):

    def draw_shape(self, image, shape, dims, color):
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
        return image
    
    def load_image(self, image_id):
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)



    def load_shapes(self, count, height, width):

        self.add_class('shapes', 1, 'square')
        self.add_class('shapes', 2, 'circle')
        self.add_class('shapes', 3, 'triangle')


        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)



    def random_shape(self, height, width):
        
        shape = random.choice(["square", "circle", "triangle"])
            # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
            # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
            # Size
        s = random.randint(buffer, height//4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y-s, x-s, y+s, x+s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes



class ShapeConfig(Config):

    NAME = 'shapes'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 3

    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    RPN_ANCHOR_SCALES = (8,16,32,64,128)

    TRAIN_ROIS_PER_IMAGE = 32 

    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 5 