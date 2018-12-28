import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.model import log
from workspace import evaluate
from workspace import helper

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class CarPlateConfig(Config):


    NAME = 'carplate'

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 33

    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.5

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    BACKBONE_STRIDES = [4, 8, 16, 24, 48]
    RPN_ANCHOR_RATIOS = [0.3, 0.6, 1]
    
    RPN_TRAIN_ANCHORS_PER_IMAGE = 500

    # RPN_NMS_THRESHOLD = 0.6
    
    IMAGE_MIN_DIM = int(480)
    IMAGE_MAX_DIM = int(640)
    
    # POST_NMS_ROIS_INFERENCE = 2000

    TRAIN_ROIS_PER_IMAGE = 400

    MEAN_PIXEL = np.array([0.449122045 * 255, 0.449122045 * 255, 0.449122045 * 255 ])

    LEARNING_RATE = 0.005



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
        self.add_class("carplate", 19, "J")
        self.add_class("carplate", 20, "K")
        self.add_class("carplate", 21, "L")
        self.add_class("carplate", 22, "M")
        self.add_class("carplate", 23, "N")
        self.add_class("carplate", 24, "P")
        self.add_class("carplate", 25, "R")
        self.add_class("carplate", 26, "S")
        self.add_class("carplate", 27, "T")
        self.add_class("carplate", 28, "U")
        self.add_class("carplate", 29, "V")
        self.add_class("carplate", 30, "W")
        self.add_class("carplate", 31, "X")
        self.add_class("carplate", 32, "Y")
        self.add_class("carplate", 33, "Z")
        # self.add_class("carplate", 1, "carplate")
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
                         "J": 19,
                         "K": 20,
                         "L": 21,
                         "M": 22,
                         "N": 23,
                         "P": 24,
                         "R": 25,
                         "S": 26,
                         "T": 27,
                         "U": 28,
                         "V": 29,
                         "W": 30,
                         "X": 31,
                         "Y": 32,
                         "Z": 33
                         }
        # name_dict = {
            # "carplate": 1
        # }
        assert subset in ['train', 'val']

        dataset_dir = os.path.join(dataset_dir, subset)
        print("Dataset: ", dataset_dir)

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))

        annotations = list(annotations.values())  # don't need the dict keys

        annotations = [a for a in annotations if a['regions']]
        
        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 
            

            class_ids = [r['region_attributes']['type'] for r in a['regions']]

            name_id = [name_dict[n] for n in class_ids]
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            self.add_image(
                "carplate",
                image_id=a['filename'],  # use file name as a unique image id
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

def display_dataset(subfolder):
    dataset_train = CarPlateDataset()
    dataset_train.load_carplates(os.path.join(ROOT_DIR, subfolder), "val")
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
    dataset_test.load_carplates(os.path.join(ROOT_DIR, subfolder), "val")
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

def bounding_boxes(subfolder):
    dataset_train = CarPlateDataset()
    dataset_train.load_carplates(os.path.join(ROOT_DIR, subfolder), "val")
    dataset_train.prepare()


    # Load random image and mask.
    # image_id = random.choice(dataset_train.image_ids)
    
    for image_id in dataset_train.image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        # Compute Bounding box
        bbox = utils.extract_bboxes(mask)


        # Display image and additional stats
        print("image_id ", image_id, dataset_train.image_reference(image_id))
    
        # Display image and instances
        visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names, figsize=(3,3))

def mini_mask(config, subfolder):
    dataset_train = CarPlateDataset()
    dataset_train.load_carplates(os.path.join(ROOT_DIR, subfolder), "val")
    dataset_train.prepare()



    image_id = np.random.choice(dataset_train.image_ids, 1)[0]
    image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
    dataset_train, config, image_id, augment=True, use_mini_mask=True)

    log("image", image)
    log("image_meta", image_meta)
    log("class_ids", class_ids)
    log("bbox", bbox)
    log("mask", mask)

    display_images([image]+[mask[:,:,i] for i in range(min(mask.shape[-1], 7))])

    visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names)

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def display_anchor(config, subfolder):

    dataset_train = CarPlateDataset()
    dataset_train.load_carplates(os.path.join(ROOT_DIR, subfolder), "val")
    dataset_train.prepare()


    backbone_shapes = modellib.compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, 
                                            config.RPN_ANCHOR_RATIOS,
                                            backbone_shapes,
                                            config.BACKBONE_STRIDES, 
                                            config.RPN_ANCHOR_STRIDE)

    # Print summary of anchors
    num_levels = len(backbone_shapes)
    anchors_per_cell = len(config.RPN_ANCHOR_RATIOS)
    print("Count: ", anchors.shape[0])
    print("Scales: ", config.RPN_ANCHOR_SCALES)
    print("ratios: ", config.RPN_ANCHOR_RATIOS)
    print("Anchors per Cell: ", anchors_per_cell)
    print("Levels: ", num_levels)
    anchors_per_level = []
    for l in range(num_levels):
        num_cells = backbone_shapes[l][0] * backbone_shapes[l][1]
        anchors_per_level.append(anchors_per_cell * num_cells // config.RPN_ANCHOR_STRIDE**2)
        print("Anchors in Level {}: {}".format(l, anchors_per_level[l]))


    image_id = np.random.choice(dataset_train.image_ids, 1)[0]
    image, image_meta, _, _, _ = modellib.load_image_gt(dataset_train, config, image_id)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    levels = len(backbone_shapes)

    for level in range(levels):
        colors = visualize.random_colors(levels)
        # Compute the index of the anchors at the center of the image
        level_start = sum(anchors_per_level[:level]) # sum of anchors of previous levels
        level_anchors = anchors[level_start:level_start+anchors_per_level[level]]
        print("Level {}. Anchors: {:6}  Feature map Shape: {}".format(level, level_anchors.shape[0], 
                                                                    backbone_shapes[level]))
        center_cell = backbone_shapes[level] // 2
        center_cell_index = (center_cell[0] * backbone_shapes[level][1] + center_cell[1])
        level_center = center_cell_index * anchors_per_cell 
        center_anchor = anchors_per_cell * (
            (center_cell[0] * backbone_shapes[level][1] / config.RPN_ANCHOR_STRIDE**2) \
            + center_cell[1] / config.RPN_ANCHOR_STRIDE)
        level_center = int(center_anchor)

        # Draw anchors. Brightness show the order in the array, dark to bright.
        for i, rect in enumerate(level_anchors[level_center:level_center+anchors_per_cell]):
            y1, x1, y2, x2 = rect
            p = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, facecolor='none',
                                edgecolor=(i+1)*np.array(colors[level]) / anchors_per_cell)
            ax.add_patch(p)
    # plt.show()

    random_rois = 2000
    g = modellib.data_generator(
        dataset_train, config, shuffle=True, random_rois=random_rois, 
        batch_size=4,
        detection_targets=True)

    # Get Next Image
    if random_rois:
        [normalized_images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, rpn_rois, rois], \
        [mrcnn_class_ids, mrcnn_bbox, mrcnn_mask] = next(g)
        
        log("rois", rois)
        log("mrcnn_class_ids", mrcnn_class_ids)
        log("mrcnn_bbox", mrcnn_bbox)
        log("mrcnn_mask", mrcnn_mask)
    else:
        [normalized_images, image_meta, rpn_match, rpn_bbox, gt_boxes, gt_masks], _ = next(g)
        
    log("gt_class_ids", gt_class_ids)
    log("gt_boxes", gt_boxes)
    log("gt_masks", gt_masks)
    log("rpn_match", rpn_match, )
    log("rpn_bbox", rpn_bbox)
    image_id = modellib.parse_image_meta(image_meta)["image_id"][0]
    print("image_id: ", image_id, dataset_train.image_reference(image_id))

    # Remove the last dim in mrcnn_class_ids. It's only added
    # to satisfy Keras restriction on target shape.
    mrcnn_class_ids = mrcnn_class_ids[:,:,0]

    b = 0

    # Restore original image (reverse normalization)
    sample_image = modellib.unmold_image(normalized_images[b], config)

    # Compute anchor shifts.
    indices = np.where(rpn_match[b] == 1)[0]
    refined_anchors = utils.apply_box_deltas(anchors[indices], rpn_bbox[b, :len(indices)] * config.RPN_BBOX_STD_DEV)
    log("anchors", anchors)
    log("refined_anchors", refined_anchors)

    # Get list of positive anchors
    positive_anchor_ids = np.where(rpn_match[b] == 1)[0]
    print("Positive anchors: {}".format(len(positive_anchor_ids)))
    negative_anchor_ids = np.where(rpn_match[b] == -1)[0]
    print("Negative anchors: {}".format(len(negative_anchor_ids)))
    neutral_anchor_ids = np.where(rpn_match[b] == 0)[0]
    print("Neutral anchors: {}".format(len(neutral_anchor_ids)))

    # ROI breakdown by class
    for c, n in zip(dataset_train.class_names, np.bincount(mrcnn_class_ids[b].flatten())):
        if n:
            print("{:23}: {}".format(c[:20], n))

    # Show positive anchors
    # fig, ax = plt.subplots(1, figsize=(16, 16))
    visualize.draw_boxes(sample_image, boxes=anchors[positive_anchor_ids], 
                        refined_boxes=refined_anchors, ax=ax)
    
    visualize.draw_boxes(sample_image, boxes=anchors[negative_anchor_ids])

    if random_rois:
        # Class aware bboxes
        bbox_specific = mrcnn_bbox[b, np.arange(mrcnn_bbox.shape[1]), mrcnn_class_ids[b], :]

        # Refined ROIs
        refined_rois = utils.apply_box_deltas(rois[b].astype(np.float32), bbox_specific[:,:4] * config.BBOX_STD_DEV)

        # Class aware masks
        mask_specific = mrcnn_mask[b, np.arange(mrcnn_mask.shape[1]), :, :, mrcnn_class_ids[b]]

        visualize.draw_rois(sample_image, rois[b], refined_rois, mask_specific, mrcnn_class_ids[b], dataset_train.class_names)
        
        # Any repeated ROIs?
        rows = np.ascontiguousarray(rois[b]).view(np.dtype((np.void, rois.dtype.itemsize * rois.shape[-1])))
        _, idx = np.unique(rows, return_index=True)
        print("Unique ROIs: {} out of {}".format(len(idx), rois.shape[1]))

    if random_rois:
    # Dispalay ROIs and corresponding masks and bounding boxes
        ids = random.sample(range(rois.shape[1]), 8)

        images = []
        titles = []
        for i in ids:
            image = visualize.draw_box(sample_image.copy(), rois[b,i,:4].astype(np.int32), [255, 0, 0])
            image = visualize.draw_box(image, refined_rois[i].astype(np.int64), [0, 255, 0])
            images.append(image)
            titles.append("ROI {}".format(i))
            images.append(mask_specific[i] * 255)
            titles.append(dataset_train.class_names[mrcnn_class_ids[b,i]][:20])

        display_images(images, titles, cols=4, cmap="Blues", interpolation="none")


    if random_rois:
        limit = 10
        temp_g = modellib.data_generator(
            dataset_train, config, shuffle=True, random_rois=10000, 
            batch_size=1, detection_targets=True)
        total = 0
        for i in range(limit):
            _, [ids, _, _] = next(temp_g)
            positive_rois = np.sum(ids[0] > 0)
            total += positive_rois
            print("{:5} {:5.2f}".format(positive_rois, positive_rois/ids.shape[1]))
        print("Average percent: {:.2f}".format(total/(limit*ids.shape[1])))
    plt.show()

def inspect_model(config, model_path):

    dataset_train = CarPlateDataset()
    dataset_train.load_carplates(os.path.join(ROOT_DIR, 'car_plate_data'), "test")
    dataset_train.prepare()
    
    model = modellib.MaskRCNN(mode='inference', config=config, model_dir=ROOT_DIR)
    model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    

    image_id = random.choice(dataset_train.image_ids)
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_train, config, image_id, use_mini_mask=False)
    info = dataset_train.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                        dataset_train.image_reference(image_id)))

    # Generate RPN trainig targets
    # target_rpn_match is 1 for positive anchors, -1 for negative anchors
    # and 0 for neutral anchors.
    target_rpn_match, target_rpn_bbox = modellib.build_rpn_targets(
        image.shape, model.get_anchors(image.shape), gt_class_id, gt_bbox, model.config)
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
    
    # visualize.draw_boxes(image, boxes=positive_anchors, refined_boxes=refined_anchors, ax=get_ax())


        # Run RPN sub-graph
    pillar = model.keras_model.get_layer("ROI").output  # node to start searching from

    # TF 1.4 and 1.9 introduce new versions of NMS. Search for all names to support TF 1.3~1.10
    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")
    if nms_node is None:
        nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
    if nms_node is None: #TF 1.9-1.10
        nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")

    rpn = model.run_graph([image], [
        ("rpn_class", model.keras_model.get_layer("rpn_class").output),
        ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
        ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
        ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
        ("post_nms_anchor_ix", nms_node),
        ("proposals", model.keras_model.get_layer("ROI").output),
    ])

    limit = 100
    sorted_anchor_ids = np.argsort(rpn['rpn_class'][:,:,1].flatten())[::-1]
    visualize.draw_boxes(image, boxes=model.anchors[sorted_anchor_ids[:limit]], ax=get_ax())

    limit = 50
    ax = get_ax(1, 2)
    pre_nms_anchors = utils.denorm_boxes(rpn["pre_nms_anchors"][0], image.shape[:2])
    refined_anchors = utils.denorm_boxes(rpn["refined_anchors"][0], image.shape[:2])
    refined_anchors_clipped = utils.denorm_boxes(rpn["refined_anchors_clipped"][0], image.shape[:2])
    visualize.draw_boxes(image, boxes=pre_nms_anchors[:limit],
                        refined_boxes=refined_anchors[:limit], ax=ax[0])
    visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[:limit], ax=ax[1])

    limit = 50
    ixs = rpn["post_nms_anchor_ix"][:limit]
    visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[ixs], ax=get_ax())

        # Show final proposals
    # These are the same as the previous step (refined anchors 
    # after NMS) but with coordinates normalized to [0, 1] range.
    limit = 50
    # Convert back to image coordinates for display
    h, w = config.IMAGE_SHAPE[:2]
    proposals = rpn['proposals'][0, :limit] * np.array([h, w, h, w])
    visualize.draw_boxes(image, refined_boxes=proposals, ax=get_ax())

    mrcnn = model.run_graph([image], [
    ("proposals", model.keras_model.get_layer("ROI").output),
    ("probs", model.keras_model.get_layer("mrcnn_class").output),
    ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
    ("masks", model.keras_model.get_layer("mrcnn_mask").output),
    ("detections", model.keras_model.get_layer("mrcnn_detection").output),
        ])
    
    # # Get detection class IDs. Trim zero padding.
    # det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
    # det_count = np.where(np.array(det_class_ids) == 0)
    # print(det_count)
    # det_class_ids = det_class_ids[:det_count]
    # detections = mrcnn['detections'][0, :det_count]

    # print("{} detections: {}".format(
    #     det_count, np.array(dataset_train.class_names)[det_class_ids]))

    # captions = ["{} {:.3f}".format(dataset_train.class_names[int(c)], s) if c > 0 else ""
    #             for c, s in zip(detections[:, 4], detections[:, 5])]
    # visualize.draw_boxes(
    #     image, 
    #     refined_boxes=utils.denorm_boxes(detections[:, :4], image.shape[:2]),
    #     visibilities=[2] * len(detections),
    #     captions=captions, title="Detections",
    #     ax=get_ax())


        # Proposals are in normalized coordinates. Scale them
    # to image coordinates.
    h, w = config.IMAGE_SHAPE[:2]
    proposals = np.around(mrcnn["proposals"][0] * np.array([h, w, h, w])).astype(np.int32)

    # Class ID, score, and mask per proposal
    roi_class_ids = np.argmax(mrcnn["probs"][0], axis=1)
    roi_scores = mrcnn["probs"][0, np.arange(roi_class_ids.shape[0]), roi_class_ids]
    roi_class_names = np.array(dataset_train.class_names)[roi_class_ids]
    roi_positive_ixs = np.where(roi_class_ids > 0)[0]

    # How many ROIs vs empty rows?
    print("{} Valid proposals out of {}".format(np.sum(np.any(proposals, axis=1)), proposals.shape[0]))
    print("{} Positive ROIs".format(len(roi_positive_ixs)))

    # Class counts
    print(list(zip(*np.unique(roi_class_names, return_counts=True))))


        # Display a random sample of proposals.
    # Proposals classified as background are dotted, and
    # the rest show their class and confidence score.
    limit = 200
    ixs = np.random.randint(0, proposals.shape[0], limit)
    captions = ["{} {:.3f}".format(dataset_train.class_names[c], s) if c > 0 else ""
                for c, s in zip(roi_class_ids[ixs], roi_scores[ixs])]
    visualize.draw_boxes(image, boxes=proposals[ixs],
                        visibilities=np.where(roi_class_ids[ixs] > 0, 2, 1),
                        captions=captions, title="ROIs Before Refinement",
                        ax=get_ax())
        # Class-specific bounding box shifts.
    roi_bbox_specific = mrcnn["deltas"][0, np.arange(proposals.shape[0]), roi_class_ids]
    log("roi_bbox_specific", roi_bbox_specific)

    # Apply bounding box transformations
    # Shape: [N, (y1, x1, y2, x2)]
    refined_proposals = utils.apply_box_deltas(
        proposals, roi_bbox_specific * config.BBOX_STD_DEV).astype(np.int32)
    log("refined_proposals", refined_proposals)

    # Show positive proposals
    # ids = np.arange(roi_boxes.shape[0])  # Display all
    limit = 5
    ids = np.random.randint(0, len(roi_positive_ixs), limit)  # Display random sample
    captions = ["{} {:.3f}".format(dataset_train.class_names[c], s) if c > 0 else ""
                for c, s in zip(roi_class_ids[roi_positive_ixs][ids], roi_scores[roi_positive_ixs][ids])]
    visualize.draw_boxes(image, boxes=proposals[roi_positive_ixs][ids],
                        refined_boxes=refined_proposals[roi_positive_ixs][ids],
                        visibilities=np.where(roi_class_ids[roi_positive_ixs][ids] > 0, 1, 0),
                        captions=captions, title="ROIs After Refinement",
                        ax=get_ax())
    
        # Remove boxes classified as background
    keep = np.where(roi_class_ids > 0)[0]
    print("Keep {} detections:\n{}".format(keep.shape[0], keep))

        # Remove low confidence detections
    keep = np.intersect1d(keep, np.where(roi_scores >= config.DETECTION_MIN_CONFIDENCE)[0])
    print("Remove boxes below {} confidence. Keep {}:\n{}".format(
        config.DETECTION_MIN_CONFIDENCE, keep.shape[0], keep))


        # Apply per-class non-max suppression
    pre_nms_boxes = refined_proposals[keep]
    pre_nms_scores = roi_scores[keep]
    pre_nms_class_ids = roi_class_ids[keep]

    nms_keep = []
    for class_id in np.unique(pre_nms_class_ids):
        # Pick detections of this class
        ixs = np.where(pre_nms_class_ids == class_id)[0]
        # Apply NMS
        class_keep = utils.non_max_suppression(pre_nms_boxes[ixs], 
                                                pre_nms_scores[ixs],
                                                config.DETECTION_NMS_THRESHOLD)
        # Map indicies
        class_keep = keep[ixs[class_keep]]
        nms_keep = np.union1d(nms_keep, class_keep)
        print("{:22}: {} -> {}".format(dataset_train.class_names[class_id][:20], 
                                    keep[ixs], class_keep))

    keep = np.intersect1d(keep, nms_keep).astype(np.int32)
    print("\nKept after per-class NMS: {}\n{}".format(keep.shape[0], keep))


        # Show final detections
    ixs = np.arange(len(keep))  # Display all
    # ixs = np.random.randint(0, len(keep), 10)  # Display random sample
    captions = ["{} {:.3f}".format(dataset_train.class_names[c], s) if c > 0 else ""
                for c, s in zip(roi_class_ids[keep][ixs], roi_scores[keep][ixs])]
    visualize.draw_boxes(
        image, boxes=proposals[keep][ixs],
        refined_boxes=refined_proposals[keep][ixs],
        visibilities=np.where(roi_class_ids[keep][ixs] > 0, 1, 0),
        captions=captions, title="Detections after NMS",
        ax=get_ax())


    plt.show()

def detection(model, image_path=None, video_path=None):
    assert image_path or video_path
    # class_names = ['BG','carplate']
    class_names = ['BG','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                            'J', 'K', 'L', 'M', 'N', 'P',  'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        imgs = os.listdir(image_path)
        for img in imgs:
            print("Running on {}".format(img))
            # Read image
            image = skimage.io.imread(image_path + '/' + img)
            # Detect objects
            # image = skimage.color.gray2rgb(image)
            r = model.detect([image], verbose=1)[0]
           
            result = helper.get_char_result(image, r['rois'], r['masks'], r['class_ids'],
                                      class_names, r['scores'], show_bbox=True, score_threshold=0.50,
                                      show_mask=False)
            print(result)
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'], figsize=(3,3), show_mask=False)
            cv2.waitKey(0)
            

def train_model(model, subfolder, mode_train):
    """Train the model."""
    # Training dataset.
    
    dataset_train = CarPlateDataset()
    dataset_train.load_carplates(os.path.join(ROOT_DIR, subfolder), "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CarPlateDataset()
    dataset_val.load_carplates(os.path.join(ROOT_DIR, subfolder), "val")
    dataset_val.prepare()

    if mode_train == 'all':
        print("Training network all")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=1000,
                    layers='all')
    elif mode_train == 'heads':
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=1000,
                    layers='all')




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
       description='Train Mask R-CNN to detect carplates.')
    
    parser.add_argument('command',
                    metavar='<command>',
                    help="'train, detect, display_data, display_box, mini_mask, display_anchor'")
   
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory inference dataset')

    parser.add_argument('--dataset_train', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory training dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--mode_train', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")


    
    args = parser.parse_args()

    if args.command == 'train':
        config = CarPlateConfig()
    elif args.command == 'detect':
        config = InferenceConfig()
    
    #config.display()

    if args.command == 'display_data':
        display_dataset(args.dataset)
    if args.command == 'display_box':
        bounding_boxes(args.dataset)
    if args.command == 'mini_mask':
        mini_mask(config, args.dataset)
    if args.command == 'display_anchor':
        display_anchor(config, args.dataset)
    if args.command == 'inspect_model':
        inspect_model(config, args.weights)
    if args.command == 'eval_carplate':
        model = modellib.MaskRCNN(mode='inference', config=config, model_dir=os.path.join(ROOT_DIR, 'logs'))
        
        model_path = os.path.join(ROOT_DIR, args.weights)
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)

        evaluate.evaluate_carplate(args.dataset, model, ROOT_DIR)

    if args.command == 'eval_chars':

        lp_path = args.weights.split(',')[0]
        chars_path = args.weights.split(',')[1]

        lp_config = evaluate.EvalCarPlateConfig()
        lp_config.display()
        chars_config = evaluate.EvalCharConfig()
        chars_config.display

        lp_model = modellib.MaskRCNN(mode='inference', config=lp_config, model_dir=os.path.join(ROOT_DIR, 'logs'))
        model_path = os.path.join(ROOT_DIR, lp_path)
        print("Loading weights ", model_path)
        lp_model.load_weights(model_path, by_name=True)


        chars_model = modellib.MaskRCNN(mode='inference', config=chars_config, model_dir=os.path.join(ROOT_DIR, 'logs'))
        model_path = os.path.join(ROOT_DIR, chars_path)
        print("Loading weights ", model_path)
        chars_model.load_weights(model_path, by_name=True)

        evaluate.evaluate_numbers(args.dataset, lp_model, chars_model, ROOT_DIR)

    
    
    if args.command == 'train':
        model = modellib.MaskRCNN(mode='training', config=config, model_dir=os.path.join(ROOT_DIR, 'logs'))
        if args.mode_train == 'all':
            weights_path = os.path.join(ROOT_DIR, 'resnet50_weights.h5')
        elif args.mode_train == 'heads':
            weights_path = args.weights
     
        print("Logs: ", os.path.join(ROOT_DIR, 'logs'))
        print("Loading weights: ", weights_path)
        
        # weights_path = 'char_101.h5'
        model.load_weights(weights_path, by_name=True)

        train_model(model, args.dataset_train, args.mode_train)
    
    if args.command == 'detect':
        model = modellib.MaskRCNN(mode='inference', config=config, model_dir=os.path.join(ROOT_DIR, 'logs'))
        # model_path = model.find_last()
        model_path = os.path.join(ROOT_DIR, args.weights)
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)

        dataset = os.path.join(ROOT_DIR, args.dataset)
        detection(model, image_path=dataset,
                                video_path=None)
































