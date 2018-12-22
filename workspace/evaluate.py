import os
import skimage.draw
import datetime
import sys
import cv2
import numpy as np 

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import visualize
from workspace import helper


class EvalCharConfig(Config):



    NAME = 'chars'

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 33

    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.5

    RPN_ANCHOR_SCALES = (32, 56, 72, 96, 128)

    RPN_ANCHOR_RATIOS = [0.3, 0.6, 1]

    RPN_TRAIN_ANCHORS_PER_IMAGE = 500

    RPN_NMS_THRESHOLD = 0.6
    
    IMAGE_MIN_DIM = int(480)
    IMAGE_MAX_DIM = int(640)
    
    POST_NMS_ROIS_INFERENCE = 2000

    TRAIN_ROIS_PER_IMAGE = 400

    MEAN_PIXEL = np.array([0.449122045 * 255, 0.449122045 * 255, 0.449122045 * 255 ])

    LEARNING_RATE = 0.005

class EvalCarPlateConfig(Config):


    NAME = 'carplate'

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1

    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.9

    # RPN_ANCHOR_SCALES = (16, 32, 48, 64, 128)

    # RPN_ANCHOR_RATIOS = [0.2, 0.5, 1]
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 320
    # IMAGE_MIN_DIM = int(480)
    # IMAGE_MAX_DIM = int(640)







def evaluate_carplate(img_folder, model, root_dir):
    now = datetime.datetime.now()
    if not os.path.exists(root_dir + "/eval_results"):
        os.mkdir(root_dir + "/eval_results" )

    eval_path = root_dir + '/eval_results/' + 'plate_' + now.isoformat()
    eval_path_txt = open(eval_path, 'w')

    check_file = root_dir + '/eval_real_data/' + 'eval.txt'
    
    name_check = {}
    with open(check_file) as fp:
        for cnt, line in enumerate(fp):
            val,key = line.split(" ")[0],  line.split(" ")[1].split("\n")[0] 
            name_check[val] = key
    class_names = ["BG, carplate"]
    correct = 0
    error = 0

    if img_folder:

        imgs = os.listdir(img_folder)
        for img in imgs:
            image = skimage.io.imread(img_folder + '/' + img)

            r = model.detect([image], verbose=1)[0]

            if  len(r['scores']) != 0 and name_check[img] == 'True':
                 eval_path_txt.write(img + '  ' + 'Correct' + '\n')
                 correct += 1
            elif len(r['scores']) != 0 and name_check[img] == 'False' :
                 eval_path_txt.write(img + '  ' + 'Error' + '\n')
                 error += 1
            elif len(r['scores']) == 0 and name_check[img] == 'True':
                 eval_path_txt.write(img + '  ' + 'Error' + '\n')
                 error += 1
            elif len(r['scores']) == 0 and name_check[img] == 'False':
                 eval_path_txt.write(img + '  ' + 'Correct' + '\n')
                 correct += 1



            
            print(r['class_ids'], r['scores'])
        print(("The total accuracy is: " +  str(int((correct/len(imgs)) * 100)) + '% '))
        print(('Total error is: ' + str(int((error/len(imgs)) * 100)) + '% '))
        print('Images accurate: ' + str(correct))
        print('Images failed: ' + str(error))
        print('Total images processed: ' + str(correct + error))
        eval_path_txt.write("The total accuracy is: " +  str(int((correct/len(imgs)) * 100)) + '% ' +'\n' )
        eval_path_txt.write('Images accurate: ' + str(correct) +'\n')
        eval_path_txt.write('Images failed: ' + str(error) +'\n')
        eval_path_txt.write('Total images processed: ' + str(correct + error) +'\n')



def evaluate_numbers(img_folder, lp_model, chars_model, root_dir):
    now = datetime.datetime.now()
    if not os.path.exists(root_dir + "/eval_results"):
        os.mkdir(root_dir + "/eval_results" )

    eval_path = root_dir + '/eval_results/' + 'chars_' + now.isoformat()
    eval_path_txt = open(eval_path, 'w')
    
    padding_carplate = [10,10]
    class_names_chars = ['BG','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                            'J', 'K', 'L', 'M', 'N', 'P',  'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    class_names_plate = ['BG', 'carplate']
    correct = 0
    error = 0
    if img_folder:

        imgs = os.listdir(img_folder)

        for img in imgs:
            image = skimage.io.imread(img_folder + '/' + img)

            r = lp_model.detect([image], verbose=1)[0]

            if len(r['rois']) != 0:
                carplate_roi = r['rois'][0]
                plate_image = image[carplate_roi[0]- padding_carplate[0]:carplate_roi[2] + padding_carplate[0],carplate_roi[1]-padding_carplate[1]:carplate_roi[3]+padding_carplate[1]]
                
                if len(plate_image) == 0:
                    continue
                c = chars_model.detect([plate_image], verbose=1)[0]

                result = helper.get_char_result(image, c['rois'], c['masks'], c['class_ids'],
                                      class_names_chars, c['scores'], show_bbox=True, score_threshold=0.50,
                                      show_mask=False)
                img_name = str(img).split('.')[0].split("_")
                if result is None:
                    error +=1
                    eval_path_txt.write(img  + ' No characters found ' + '\n')
                    continue

                res_img = result.split("\n")
                print("Image name: ", img_name, " Detected number: ", res_img)
                if res_img[0] == '' and len(img_name[0]) != 2:
                    if res_img[1] == img_name[0]:
                        print('Match one line')
                        correct += 1
                        eval_path_txt.write(img_name[0] + '  ' + ' Match one line ' + res_img[1] + '\n')
                    else:
                        print('No match one line')
                        error +=1
                        eval_path_txt.write(img_name[0] + '  ' + ' No match one line ' + res_img[1] + '\n')

                if res_img[0] != '':
                    if res_img[0] == img_name[0] and res_img[1] == img_name[1]:
                        print('Match two lines')
                        correct +=1
                        eval_path_txt.write(img_name[0] + img_name[1] + '  ' + ' Match two lines ' + res_img[0] + res_img[1] + '\n')
                    else:
                        print('No match two lines')  
                        error += 1
                        if len(img_name) == 2:
                            eval_path_txt.write(img_name[0] + img_name[1] + '  ' + ' No match two lines ' + res_img[0] + res_img[1] + '\n')
                        elif len(img_name) == 1:
                            eval_path_txt.write(img_name[0] + '  ' + ' No match two lines ' + res_img[0] + res_img[1] + '\n')
                if res_img[0] == '' and len(img_name[0]) == 2:
                    print('Detected one line, but there two lines')
                    error += 1
                    eval_path_txt.write(img  + ' Detected one line, but there two lines ' + res_img[1] + '\n')
                    
                
                # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                #                 class_names_plate, r['scores'], figsize=(3,3), show_mask=False)
                # visualize.display_instances(plate_image, c['rois'], c['masks'], c['class_ids'], 
                #                     class_names_chars, figsize=(3,3), show_mask=False, title=result, show_bbox=False)
            else:
                eval_path_txt.write(img + '  ' + ' No carplate found ' + '\n')
                print("No carplate found")


        print(("The total accuracy is: " +  str(int((correct/len(imgs)) * 100)) + '% '))
        print(('Total error is: ' + str(int((error/len(imgs)) * 100)) + '% '))
        print('Images accurate: ' + str(correct))
        print('Images failed: ' + str(error))
        print('Total images processed: ' + str(correct + error))
        eval_path_txt.write("The total accuracy is: " +  str(int((correct/len(imgs)) * 100)) + '% ' +'\n' )
        eval_path_txt.write('Images accurate: ' + str(correct) +'\n')
        eval_path_txt.write('Images failed: ' + str(error) +'\n')
        eval_path_txt.write('Total images processed: ' + str(correct + error) +'\n')
          


def create_eval_dataset(root_dir, file, img_folder):
    if not os.path.exists(root_dir + "/eval_real_data"):
        os.mkdir(root_dir + "/eval_real_data" )

    eval_path = root_dir + '/eval_real_data' + '/' + file
    eval_path_txt = open(eval_path, 'w')
    if img_folder:

        imgs = os.listdir(img_folder)
        imgs.sort(key=lambda x: int(os.path.splitext(x)[0]))
        for img in imgs:
            # image = skimage.io.imread(img_folder+'/'+img)
            eval_path_txt.write(img + '\n')


