import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class CarPlateConfig(Config):


    NAME = 'carplate'

    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 37

    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.9

class InferenceConfig(CarPlateConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class CarPlateDataset(utils.Dataset):

   def load_carplates(self, dataset_dir, subset):

        self.add_class("carplate", 1, "0")
        self.add_class("carplate", 2, "1")
        self.add_class("carplate", 3, "2")
        self.add_class("carplate", 4, "3")
        self.add_class("carplate", 5, "4")
        self.add_class("carplate", 6, "5")
        self.add_class("carplate", 7, "6")
        self.add_class("carplate", 8, "7")
        self.add_class("carplate", 9, "8")
        self.add_class("carplate", 10, "9")
        self.add_class("carplate", 11, "A")
        self.add_class("carplate", 12, "B")
        self.add_class("carplate", 13, "C")
        self.add_class("carplate", 14, "D")
        self.add_class("carplate", 15, "E")
        self.add_class("carplate", 16, "F")
        self.add_class("carplate", 17, "G")
        self.add_class("carplate", 18, "H")
        self.add_class("carplate", 19, "I")
        self.add_class("carplate", 20, "J")
        self.add_class("carplate", 21, "K")
        self.add_class("carplate", 22, "L")
        self.add_class("carplate", 23, "M")
        self.add_class("carplate", 24, "N")
        self.add_class("carplate", 25, "O")
        self.add_class("carplate", 26, "P")
        self.add_class("carplate", 27, "Q")
        self.add_class("carplate", 28, "R")
        self.add_class("carplate", 29, "S")
        self.add_class("carplate", 30, "T")
        self.add_class("carplate", 31, "U")
        self.add_class("carplate", 32, "V")
        self.add_class("carplate", 33, "W")
        self.add_class("carplate", 34, "X")
        self.add_class("carplate", 35, "Y")
        self.add_class("carplate", 36, "Z")
        self.add_class("carplate", 37, "car_plate")


        name_dict = {"0": 1,
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
                         "I": 19,
                         "J": 20,
                         "K": 21,
                         "L": 22,
                         "M": 23,
                         "N": 24,
                         "O": 25,
                         "P": 26,
                         "Q": 27,
                         "R": 28,
                         "S": 29,
                         "T": 30,
                         "U": 31,
                         "V": 32,
                         "W": 33,
                         "X": 34,
                         "Y": 35,
                         "Z": 36,
                         "car_plate": 37

                         }

        assert subset in ['train', 'test']

        dataset_dir = os.path.join(dataset_dir, subset)
        file_json = ''
        if subset == 'train':
                file_json = 'general_train.json'
        else:
                file_json = 'general_test.json'
        annotations = json.load(open(os.path.join(dataset_dir, file_json)))

        img_keys = list(annotations['_via_img_metadata'])

        for key in img_keys:
            if type(annotations['_via_img_metadata'][key]['regions']) is list:
                polygons = [r['shape_attributes'] for r in annotations['_via_img_metadata'][key]['regions']]

            class_ids = [r['region_attributes']['type'] for r in annotations['_via_img_metadata'][key]['regions']]
            name_id = [name_dict[n] for n in class_ids]
            image_path = os.path.join(dataset_dir, annotations['_via_img_metadata'][key]['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            self.add_image(
                "carplate",
                image_id=annotations['_via_img_metadata'][key]['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                class_id = name_id)

   def load_mask(self, image_id):
   
        image_info = self.image_info[image_id]
        if image_info["source"] != "carplate":
            return super(self.__class__, self).load_mask(image_id)
        name_id = image_info["class_id"]
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        class_ids = np.array(name_id, dtype=np.int32)
        for i, p in enumerate(info["polygons"]):
            if 'all_points_x' and 'all_points_y' in p:
            # Get indexes of pixels inside the polygon and set them to 1
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                mask[rr, cc, i] = 1
        
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return (mask.astype(np.bool),class_ids)

   def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "carplate":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def display_dataset():
    dataset_train = CarPlateDataset()
    dataset_train.load_carplates(os.path.join(ROOT_DIR, 'car_plate_data'), "train")
    dataset_train.prepare()

  
    print("Image Count: {}".format(len(dataset_train.image_ids)))
    print("Class Count: {}".format(dataset_train.num_classes))
    for i, info in enumerate(dataset_train.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    image_ids = np.random.choice(dataset_train.image_ids, 3)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names, limit=7)


    dataset_test = CarPlateDataset()
    dataset_test.load_carplates(os.path.join(ROOT_DIR, 'car_plate_data'), "test")
    dataset_test.prepare()

  
    print("Image Count: {}".format(len(dataset_test.image_ids)))
    print("Class Count: {}".format(dataset_test.num_classes))
    for i, info in enumerate(dataset_test.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    image_ids = np.random.choice(dataset_test.image_ids, 3)
    for image_id in image_ids:
        image = dataset_test.load_image(image_id)
        mask, class_ids = dataset_test.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_test.class_names, limit=7)



def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash
def display_mask_image(cv_window_name,image, boxes, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None,
                       score_threshold=0.8):
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
    if not N:
        print("\n*** No instances to display *** \n")
        return image
    else:
        assert boxes.shape[0] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = True
    # if not ax:
    #     _, ax = plt.subplots(1, figsize=figsize)
    #     auto_show = True
    #
    # Generate random colors
    # colors = visualize.fixed_colors(N)
    colors = visualize.random_colors(N)
    # Show area outside image boundaries.
    # height, width = image.shape[:2]
    # ax.set_ylim(height + 10, -10)
    # ax.set_xlim(-10, width + 10)
    # ax.axis('off')
    # ax.set_title(title)
    scoreMin=score_threshold
    masked_image=image.astype(np.uint8).copy()
    for i in range(N):
        print('class:',class_names[class_ids[i]],'score:',scores[i])
        if scores[i] > scoreMin:
            # Mask
            # mask = masks[:, :, i]
            # color = colors[i]
            # color = colors[class_ids[i]]
            color = colors[0]
            # color =
            # if show_mask:
                # masked_image = visualize.apply_mask(image, mask, color,alpha=0.4)

    masked_image = masked_image.astype(np.uint8).copy()
    for i in range(N):
        if scores[i]>scoreMin:
            # color = colors[i]
            # color = colors[class_ids[i]]
            color = colors[0]
            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            if show_bbox:
                image = cv2.rectangle(masked_image, (x1, y1),(x2, y2),(100,20,100),thickness=2)
                # p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                #                     alpha=0.7, linestyle="dashed",
                #                     edgecolor=color, facecolor='none')
                # ax.add_patch(p)

            # Label
            if not captions:
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                # x = random.randint(x1, (x1 + x2) // 2)
                # caption = "{}  {:.3f}".format(label, score) if score else label
                caption = "{}  ".format(label) if score else label
            else:
                caption = captions[i]

            image = cv2.addText(image,caption, (x1, y1-1),'Times',pointSize=13,color=(178,34,34))
            # ax.text(x1, y1 + 8, caption,
            #         color='w', size=11, backgroundcolor="none")


            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            # padded_mask = np.zeros(
            #     (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            # padded_mask[1:-1, 1:-1] = mask
            # contours = visualize.find_contours(padded_mask, 0.5)
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
    return masked_image
    
def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path
    class_names = ['BG','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                           'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'car_plate']
    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        imgs = os.listdir(image_path)
        for img in imgs:
            print("Running on {}".format(img))
            # Read image
            image = skimage.io.imread(image_path + '/' + img)
            # Detect objects
            image = skimage.color.gray2rgb(image)
            r = model.detect([image], verbose=1)[0]

            cv_window_name = "Mask-RCNN for car plate"
            cv2.namedWindow(cv_window_name)

            mask_img = display_mask_image(cv_window_name, image, r['rois'], r['class_ids'],
                                          class_names, r['scores'], show_bbox=True,score_threshold=0.9,show_mask=False)
           
            cv2.imshow(cv_window_name, mask_img)
            k = cv2.waitKey(0)
            if k == 27:
                continue
            # Color splash
            # splash = color_splash(image, r['masks'])
            # Save output
            # file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
            # skimage.io.imsave(file_name, splash)
    elif video_path:
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                # image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CarPlateDataset()
    dataset_train.load_carplates(os.path.join(ROOT_DIR, 'car_plate_data'), "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CarPlateDataset()
    dataset_val.load_carplates(os.path.join(ROOT_DIR, 'car_plate_data'), "test")
    dataset_val.prepare()


    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
       description='Train Mask R-CNN to detect carplates.')
    
    parser.add_argument('command',
                    metavar='<command>',
                    help="'train'")
    
    
    args = parser.parse_args()

    if args.command == 'display_data':
        display_dataset()

    if args.command == 'train':
        config = CarPlateConfig()
    elif args.command == 'detect':
        config = InferenceConfig()
    
    config.display()
    
    
    if args.command == 'train':
        model = modellib.MaskRCNN(mode='training', config=config, model_dir=os.path.join(ROOT_DIR, 'logs'))

        weights_path = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
        train(model)
    
    if args.command == 'detect':
        model = modellib.MaskRCNN(mode='inference', config=config, model_dir=os.path.join(ROOT_DIR, 'logs'))
        model_path = model.find_last()

        model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

        detect_and_color_splash(model, image_path='/home/dmitry/Documents/Projects/mask_rcnn_carplate/car_plate_data/test',
                                video_path=None)