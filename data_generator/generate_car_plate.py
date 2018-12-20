
__all__ = (
    'generate_ims',
)


import itertools
import math
import os
import random
import sys

import matplotlib.pyplot as plt 

import cv2
import numpy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import skimage
from skimage import measure, data, color
from skimage.draw import polygon_perimeter
import json


from keras.preprocessing.image import *


import common

ROOT_DIR = os.path.abspath("../")


FONT_DIR = "./fonts"

FONT_HEIGHT = 22

OUTPUT_SHAPE = (60,120)

CHARS = common.CHARS + " "


def make_char_ims(font_path, choose_color):
    font_size = FONT_HEIGHT * 4
    font = ImageFont.truetype(font_path, font_size)
    height = max(font.getsize(c)[1] for c in CHARS)
    # color_distrib=[180,255]
    # choose_color= color_distrib[random.randint(0,1)]
    for c in CHARS:
        width = font.getsize(c)[0]#row
        im = Image.new("RGBA", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(im)
        if './fonts/vagrounded-bold.ttf' == font_path:
            draw.text((0, -20), c, (choose_color, choose_color, choose_color), font=font)
        elif './fonts/UKNumberPlate.ttf' == font_path:
            draw.text((0, 0), c, (choose_color, choose_color, choose_color), font=font)
        scale = float(FONT_HEIGHT) / height
        im = im.resize((int(width * scale), FONT_HEIGHT), Image.ANTIALIAS)
        yield c, numpy.array(im)[:, :, 0].astype(numpy.float32) / 255.


def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M


def pick_colors():
    pick = random.randint(0,1)
    if pick == 0:
        plate_color_pick = random.randint(180,250)
        text_color = random.randint(10,30)
    else:
        plate_color_pick = random.randint(150,250)
        text_color = random.randint(10,30)

    return text_color/255, plate_color_pick/255

def pick_font_hight():
    return random.choice([22,50])


def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                              numpy.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= numpy.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (numpy.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if numpy.any(trans < -0.5) or numpy.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = numpy.hstack([M, trans + center_to - M * center_from])


    return M, out_of_bounds

def generate_code_two_line():
    firstline= " {}{} ".format( random.choice(common.LETTERS), random.choice(common.LETTERS))
    secondline="{}{}{}{}".format( random.choice(common.DIGITS), random.choice(common.DIGITS), random.choice(common.DIGITS), random.choice(common.DIGITS))
    return  firstline,secondline
def generate_code():
    input = []
    for i in range(7):
        if random.randint(0,1)==0:
            input.append( random.choice(common.LETTERS))
        else:
            input.append(random.choice(common.DIGITS))
    return "{}{} {}{}{}{}".format(input[0],input[1],input[2],input[3],input[4],input[5],input[6])

def rounded_rect(shape, radius):
    out = numpy.ones(shape)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)

    return out


def generate_plate(font_height, char_ims, font_name):
    if font_name == 'vagrounded-bold.ttf':

        h_padding = random.uniform(0.01, 0.02) * font_height
        v_padding = random.uniform(0.001, 0.002) * font_height
        spacing = font_height * random.uniform(-0.01, 0.01)
        radius = 1 + int(font_height * 0.1 * random.random())

    if font_name == 'UKNumberPlate.ttf':

        h_padding = 0.4 * font_height
        v_padding = 0.1 * font_height
        spacing = font_height * random.uniform(-0.01, 0.01)
        radius = 1 + int(font_height * 0.1 * random.random())

    code = generate_code()

    all = code
    max = 0
    for c in all:
        if char_ims[c].shape[1] > max:
            max = char_ims[c].shape[1]

    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    if font_name == 'vagrounded-bold.ttf':
        out_shape = (int(font_height + v_padding * 4),
                    int(text_width + h_padding * 1))
    if font_name == 'UKNumberPlate.ttf':
        out_shape = (int(font_height + v_padding * 3),
                    int(text_width + h_padding * 1))

  
    text_color, plate_color = pick_colors()

    
    text_mask = numpy.zeros(out_shape)
    # text_mask = text_mask * (1-text_color)
    # ig, ax = plt.subplots()

    x = h_padding
    y = v_padding
    count=0
    letters=[]
    bounding_box= []
    for c in range(len(code)):
        count+=1
        char_im = char_ims[code[c]]
        start=int(char_ims[code[c]].shape[1]/2-char_im.shape[1]/2)
        end=int(char_ims[code[c]].shape[1]/2+char_im.shape[1]/2)
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing
        #letter
        if code[c]!=' ':
            letter = numpy.zeros(out_shape)
            letter[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
            letter = letter * 255
            letters.append(letter)

            # ax.imshow(text_mask, interpolation='nearest', cmap =plt.cm.gray)
            # plt.scatter(ix+start, iy+start)
            # plt.scatter(ix+start, iy+start + char_im.shape[0])
            # plt.scatter(ix+start + char_im.shape[1], iy+start)
            # plt.scatter(ix+start + char_im.shape[1], iy+start + char_im.shape[0])

            bounding_box.append([[ix+start, iy+start], [ix+start + char_im.shape[1], iy+start], [ix+start + char_im.shape[1], iy+start + char_im.shape[0]] , [ix+start, iy+start + char_im.shape[0]]])
            
    
    plate = (numpy.ones(out_shape) * plate_color * (1. - text_mask)  +
             numpy.ones(out_shape)  * text_color * text_mask)
    
    # fig, ax = plt.subplots()
    # ax.imshow(plate, interpolation='nearest', cmap =plt.cm.gray)
    
    # for key, value in bounding_box.items():
    #     plt.plot(value['x'], value['y'], linewidth=2)
    # print(bounding_box)
    # plt.show()

    # plt.show()
    
    return plate, rounded_rect(out_shape, radius), code.replace(" ", ""), bounding_box

def generate_two_lines_plate(font_height,char_ims, font_name):

    # for vba font
    bounding_box = []
    if font_name == 'vagrounded-bold.ttf':
        h_padding = 0.0005 * font_height
        v_padding = 0.0005 * font_height
        spacing = font_height * 0.0000001

    # for UK font
    if font_name == 'UKNumberPlate.ttf':
        h_padding = 0.2 * font_height
        v_padding = 0.2 * font_height
        spacing = font_height * 0.001
    radius = 1 + int(font_height * 0.1 * random.random())



    first_code_line,second_code_line = generate_code_two_line()

    all = second_code_line + first_code_line
    max = 0
    for c in all:
        if char_ims[c].shape[1] > max:
            max = char_ims[c].shape[1]

    text_width = 4*max
    text_width += (len(second_code_line) - 1) * spacing

    out_shape = (int(font_height*2 + v_padding * 2),
                 int(text_width + h_padding * 2))

    text_color, plate_color = pick_colors()
    text_mask = numpy.zeros(out_shape)
    
    
    # ig, ax = plt.subplots()
    
    x = h_padding
    y = v_padding
    count=0
    for c in range(len(first_code_line)):
        char_im = char_ims[first_code_line[c]]
        ix, iy = int(x), int(y*0.0001)
        start=int(max/2-char_im.shape[1]/2)
        end=int(max/2+char_im.shape[1]/2)
        text_mask[iy:iy + char_im.shape[0], ix+start:ix+end] = char_im
        x += max + spacing
        # cv2.imshow("", text_mask)
        # cv2.waitKey(0)
        if first_code_line[c] != " ":
            # ax.imshow(text_mask, interpolation='nearest', cmap =plt.cm.gray)
            # plt.scatter(ix+start, start)
            # plt.scatter(ix+start, start + char_im.shape[0])
            # plt.scatter(ix+start + char_im.shape[1], start)
            # plt.scatter(ix+start + char_im.shape[1], start + char_im.shape[0])

            bounding_box.append([[ix+start, start], [ix+start + char_im.shape[1], start], [ix+start + char_im.shape[1], start + char_im.shape[0]], [ix+start, start + char_im.shape[0]]])
            
    x = h_padding
    y = v_padding

    for d in range(len(second_code_line)):
        char_im = char_ims[second_code_line[d]]
        ix, iy = int(x), int(y*0.0001+font_height)
        start = int(char_ims[second_code_line[d]].shape[1] / 2 - char_im.shape[1] / 2)
        end = int(char_ims[second_code_line[d]].shape[1]  / 2 + char_im.shape[1] / 2)
        text_mask[iy:iy + char_im.shape[0], ix+start:ix+end] = char_im
        
        x += max + spacing

        if second_code_line[d] != " ":
            # ax.imshow(text_mask, interpolation='nearest', cmap =plt.cm.gray)
            # plt.scatter(ix+start, iy+start)
            # plt.scatter(ix+start, iy+start + char_im.shape[0])
            # plt.scatter(ix+start + char_im.shape[1], iy+start)
            # plt.scatter(ix+start + char_im.shape[1], iy+start + char_im.shape[0])
            
            bounding_box.append([[ix+start, iy+start],  [ix+start + char_im.shape[1], iy+start], [ix+start + char_im.shape[1], iy+start + char_im.shape[0]], [ix+start, iy+start + char_im.shape[0]]])
    
    text_mask = text_mask.reshape((text_mask.shape[0], text_mask.shape[1],1))
    if random.randint(0,1) == 0:
        text_mask = cv2.GaussianBlur(text_mask,(7,7),0)
    text_mask = text_mask.reshape((text_mask.shape[0], text_mask.shape[1]))
   
    plate = (numpy.ones(out_shape) * plate_color * (1. - text_mask) +
             numpy.ones(out_shape) * text_color * text_mask)
    

    # fig, ax = plt.subplots()
    # ax.imshow(plate, interpolation='nearest', cmap =plt.cm.gray)
    
    # for key, value in bounding_box.items():
    #     plt.plot(value['x'], value['y'], linewidth=2)
    # print(bounding_box)
    # plt.show()
    
    code=first_code_line+second_code_line
    return plate, rounded_rect(out_shape, radius), code.replace(" ", ""), bounding_box


def generate_bg(num_bg_images):
    found = False
    length = len(num_bg_images)
    while not found:
        index = int(random.randint(0, length - 1))
        img_name = num_bg_images[index]
        fname = "bgs/"+img_name
        # bg = cv2.imread(fname)
        bg = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) / 255.
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True

    bg = cv2.resize(bg,(OUTPUT_SHAPE[1],OUTPUT_SHAPE[0]))
    

    return bg,img_name

def generate_im(char_ims, num_bg_images, font_name):
    
    bg,img_name = generate_bg(num_bg_images)
    is_plate_one = False
    if random.randint(0,1) == 0:
        plate, plate_mask, code, bounding_box = generate_two_lines_plate(FONT_HEIGHT, char_ims, font_name)
        is_plate_one = False
    else:
        plate, plate_mask, code, bounding_box = generate_plate(FONT_HEIGHT, char_ims, font_name)
        is_plate_one = True
        
    # if random.randint(0,1) == 0:
    #     plate = random_zoom(plate, (0.7,1.3))
    # if random.randint(0,1) == 0:
    plate = plate.reshape((plate.shape[0], plate.shape[1],1))
    # plate = random_brightness(plate, (0.041,0.005))
   
    plate = plate.reshape((plate.shape[0], plate.shape[1]))
    M, out_of_bounds = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            min_scale=0.75,
                            max_scale=1.0,
                            rotation_variation=0.9,
                            scale_variation=0,
                            translation_variation=0.8)
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))
    
    M_aug = numpy.zeros((M.shape[0] + 1, M.shape[1])) 
    M_aug[:2,:] = M
    M_aug[-1,-1] = 1

    json_txt = []
    contours = []
    for i in range(len(code)):
        edge = []
        all_points_x = []
        all_points_y = []
        curr_img = bounding_box[i]
        for point in curr_img:
            point = [point[0], point[1]]
            after_trans = numpy.dot(M_aug, numpy.hstack((numpy.array(point),1)).reshape(-1,1))
            y = after_trans[1]
            x = after_trans[0]

            all_points_x.append(x.tolist()[0])
            all_points_y.append(y.tolist()[0])
            edge.append([x,y])
        
        json_txt.append({'shape_attributes': {'name': 'polyline', 'all_points_x':all_points_x, 
                                                        'all_points_y':all_points_y}, 'region_attributes':{'name': code[i], 'type': code[i]}  })
        contours.append(numpy.array(edge))
    
    out = plate * plate_mask + bg * (1.0 - plate_mask)
    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    out += numpy.random.normal(scale=0.05, size=out.shape)
    out = numpy.clip(out, 0., 1.)
    out = skimage.color.gray2rgb(out)




    return plate_mask,out, code, not out_of_bounds ,json_txt

def show_contours(img, contours):
    x_points = []
    y_points = []
    for contour in contours:
        for edge in contour:
            x_points.append(edge[0])
            y_points.append(edge[1])
            print("\n")
    
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap =plt.cm.gray)

     # for n, contour in enumerate(result):
    plt.plot(x_points, y_points, linewidth=2)
    plt.show()

def load_fonts(folder_path):
    font_char_ims = {}
    color_distrib = [188,255,114,90,40,random.randint(190, 220)]
    plate_color_pick = [65,170,139,182,249,random.randint(30,90)] 
    my_list = [0] * 13 + [1] * 13 + [2] * 21 + [3] * 20 + [4] * 10 + [5] * 33
    number = random.choice(my_list)

    fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]
    for font in fonts:
        font_char_ims[font] = dict(make_char_ims(os.path.join(folder_path,
                                                              font),
                                                  color_distrib[number]))
    return fonts, font_char_ims, plate_color_pick[number]


def generate_ims():
    """
    Generate number plate images.

    :return:
        Iterable of number plate images.

    """
    variation = 1.0
    
    num_bg_images = os.listdir("bgs")
    while True:
        fonts, font_char_ims, plate_color = load_fonts(FONT_DIR)
        font_name = random.choice(fonts)
        yield generate_im(font_char_ims[font_name], num_bg_images, font_name)


def run(train_number, val_number, txt_file_train='via_region_data.json', txt_file_test='via_region_data.json', folder_train='train', folder_test='val'):


    data_path = os.path.join(ROOT_DIR, 'car_plate_data')

    if not os.path.exists( data_path + "/" + folder_train):
        os.mkdir(ROOT_DIR + "/car_plate_data/" + folder_train)
    if not os.path.exists(data_path + "/" + folder_test):
        os.mkdir(ROOT_DIR + "/car_plate_data/" + folder_test)

    
    labelPath1 = ROOT_DIR+ '/car_plate_data/' + folder_train + "/" + txt_file_train
    f_labeltxt = open(labelPath1, 'w')
    labelPath2 = ROOT_DIR+ '/car_plate_data/' + folder_test + "/" + txt_file_test
    f_labeltxt2 = open(labelPath2, 'w')

    im_gen = itertools.islice(generate_ims(), train_number)
    json_txt_train = {}
    json_txt_test = {}
    for img_idx, (plate_mask, im, c, p, label_txt) in enumerate(im_gen):
        if label_txt == '0':
            print('called')
            continue
     
        fname = ROOT_DIR + "/car_plate_data/"+ folder_train +"/{:08d}_{}_{}.png".format(img_idx, c,
                                                        "1" if p else "0")
        img_name = "{:08d}_{}_{}.png".format(img_idx, c,
                                             "1" if p else "0")
        fname2 =ROOT_DIR + "/car_plate_data/mask_plate_train/{:08d}_{}_{}.png".format(img_idx, c,
                                                         "1" if p else "0")
        print(str(img_idx) + "/" + str(train_number) + "-------" +fname)
        cv2.imwrite(fname, im * 255.)
        json_txt = {'filename': img_name, 'regions': label_txt}
        json_txt_train[img_name] = json_txt
    json.dump(json_txt_train, f_labeltxt)

    im_gen = itertools.islice(generate_ims(), val_number)
    for img_idx, (plate_mask, im, c, p, label_txt) in enumerate(im_gen):
        if label_txt == '0':
            print('called')
            continue
        fname = ROOT_DIR + "/car_plate_data/"+ folder_test +"/{:08d}_{}_{}.png".format(img_idx, c,
                                                        "1" if p else "0")
        img_name = "{:08d}_{}_{}.png".format(img_idx, c,
                                             "1" if p else "0")
        fname2 =ROOT_DIR + "/car_plate_data/mask_plate_test/{:08d}_{}_{}.png".format(img_idx, c,
                                                         "1" if p else "0")
        print(str(img_idx)+ "/" + str(val_number) + "-------"+fname)
        json_txt = {'filename': img_name, 'regions': label_txt}
        json_txt_test[img_name] = json_txt
        cv2.imwrite(fname, im * 255.)
    json.dump(json_txt_test, f_labeltxt2)

    f_labeltxt.close()
    f_labeltxt2.close()