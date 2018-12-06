#!/usr/bin/env python
#
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.



"""
Generate training and test images.

"""


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

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import *
import common

ROOT_DIR = os.path.abspath("../")

FONT_DIR = "./fonts"
FONT_HEIGHT = random.randint(10,22)  # Pixel size to which the chars are resized

OUTPUT_SHAPE = (80,170)

CHARS = common.CHARS + " "


def make_char_ims(font_path, output_height):
    font_size = output_height * 4

    font = ImageFont.truetype(font_path, font_size)

    height = max(font.getsize(c)[1] for c in CHARS)

    for c in CHARS:
        width = font.getsize(c)[0]#row
        im = Image.new("RGBA", (width, height), (0, 0, 0))

        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
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
    first = True
    while first or plate_color - text_color < 0.3:
        text_color = random.random()
        plate_color = random.randint(190,235)
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color/255


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
    #return " DD ","0101"
def generate_code():
    #ormat={"{}{}{}{} {}{}{}","{}{}{}{}{}{}{}"}
    input = []
    for i in range(7):
        if random.randint(0,1)==0:
            input.append( random.choice(common.LETTERS))
        # elif random.randint(0,2)==1:
        #     input.append(random.choice(common.DIGITS))
        else:
            input.append(random.choice(common.DIGITS))
    return "{}{} {}{}{}{}".format(input[0],input[1],input[2],input[3],input[4],input[5],input[6])
    #return "DD 0101"
    # return "{}{}{}{} {}{}{}".format(
    #     random.choice(common.LETTERS),
    #     random.choice(common.LETTERS),
    #     random.choice(common.DIGITS),
    #     random.choice(common.DIGITS),
    #     random.choice(common.LETTERS),
    #     random.choice(common.LETTERS),
    #     random.choice(common.LETTERS))


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


def generate_plate(font_height, char_ims):
    h_padding = random.uniform(0.2, 0.4) * font_height
    v_padding = random.uniform(0.1, 0.3) * font_height
    spacing = font_height * random.uniform(-0.05, 0.05)
    radius = 1 + int(font_height * 0.1 * random.random())

    code = generate_code()
    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2),
                 int(text_width + h_padding * 2))
    
  
    text_color, plate_color = pick_colors()
    
    text_mask = numpy.zeros(out_shape)

    x = h_padding
    y = v_padding
    count=0
    letters=[]
    for c in code:
        count+=1
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing

        #letter
        if c!=' ':
            letter = numpy.zeros(out_shape)
            letter[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
            letter = letter * 255
            # cv2.imwrite(c+'mask.jpg',letter)
            letters.append(letter)
            # letter1 = text_mask*255
   
    plate = (numpy.ones(out_shape) * plate_color * (1. - text_mask) +
             numpy.ones(out_shape) * 0.01 * text_mask)
    return plate, rounded_rect(out_shape, radius), code.replace(" ", "")

def generate_two_lines_plate(font_height,char_ims):
    h_padding = 0.2 * font_height
    v_padding = 0.2 * font_height
    spacing = font_height * 0.1
    radius = 1 + int(font_height * 0.1 * random.random())



    first_code_line,second_code_line = generate_code_two_line()

    all = second_code_line + first_code_line
    max = 0
    for c in all:
        if char_ims[c].shape[1] > max:
            max = char_ims[c].shape[1]
    #
    text_width = 4*max
    text_width += (len(second_code_line) - 1) * spacing
 

    out_shape = (int(font_height*2 + v_padding * 3),
                 int(text_width + h_padding * 2))

    text_color, plate_color = pick_colors()

    text_mask = numpy.zeros(out_shape)

    x = h_padding
    y = v_padding
    count=0
    for c in first_code_line:
        char_im = char_ims[c]
        ix, iy = int(x), int(y)

        start=int(max/2-char_im.shape[1]/2)
        end=int(max/2+char_im.shape[1]/2)
        text_mask[iy:iy + char_im.shape[0], ix+start:ix+end] = char_im
        #text_mask[iy:iy + char_im.shape[0], ix:(ix + max)] = char_im
        x += max + spacing
        #count+=1

    x = h_padding
    y = v_padding
    for d in second_code_line:
        char_im = char_ims[d]
        ix, iy = int(x), int(y*2+font_height)
        start = int(max / 2 - char_im.shape[1] / 2)
        end = int(max / 2 + char_im.shape[1] / 2)
        text_mask[iy:iy + char_im.shape[0], ix+start:ix+end] = char_im
        x += max + spacing


    plate = (numpy.ones(out_shape) * plate_color * (1. - text_mask) +
             numpy.ones(out_shape) * 0.01 * text_mask)
    
    

    code=first_code_line+second_code_line
    return plate, rounded_rect(out_shape, radius), code.replace(" ", "")#set the name


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

def incerease_brightness(img, value):
    img = numpy.uint8(img)
    print(img.shape)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(hsv)

    lim = 255 -value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h,s,v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def generate_im(char_ims, num_bg_images):
    bg,img_name = generate_bg(num_bg_images)

    is_plate_one = False
    if random.randint(0,1) == 0:
        
        plate, plate_mask, code = generate_two_lines_plate(FONT_HEIGHT, char_ims)
        is_plate_one = False
    else:
        plate, plate_mask, code = generate_plate(FONT_HEIGHT, char_ims)
        is_plate_one = True
    
    
    plate = plate.reshape((plate.shape[0], plate.shape[1],1))

    if random.randint(0,1) == 0:
        plate = random_zoom(plate, (0.7,1.3))
    if random.randint(0,1) == 0:
        plate = random_brightness(plate, (0.01,0.013))
    # if random.randint(0,1) == 0:
        # plate = random_shear(plate, 10)
    
    
    plate = plate.reshape((plate.shape[0], plate.shape[1]))



    M, out_of_bounds = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            min_scale=0.5,
                            max_scale=1.0,
                            rotation_variation=1.0,
                            scale_variation=0.8,
                            translation_variation=0.8)



    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    label_txt = writeContour(plate, code, is_plate_one)
    if random.randint(0,1) == 0:
        plate = cv2.blur(plate,(3,3))
    if random.randint(0,1) == 0:
        plate = cv2.GaussianBlur(plate, (5,5),0)


    letters_Affine=[]
    index = 0

    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))


    out = plate * plate_mask + bg * (1.0 - plate_mask)
    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    out += numpy.random.normal(scale=0.05, size=out.shape)
    out = numpy.clip(out, 0., 1.)
    out = skimage.color.gray2rgb(out)
    return plate_mask,out, code, not out_of_bounds ,label_txt


def load_fonts(folder_path):
    font_char_ims = {}
    fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]
    for font in fonts:
        font_char_ims[font] = dict(make_char_ims(os.path.join(folder_path,
                                                              font),
                                                 FONT_HEIGHT))
    return fonts, font_char_ims


def generate_ims():
    """
    Generate number plate images.

    :return:
        Iterable of number plate images.

    """
    variation = 1.0
    fonts, font_char_ims = load_fonts(FONT_DIR)
    num_bg_images = os.listdir("bgs")
    while True:
        yield generate_im(font_char_ims[random.choice(fonts)], num_bg_images)

def drawShape(img, coordinates, color):

    img = skimage.color.gray2rgb(img)

    coordinates = coordinates.astype(int)
    img[coordinates[:, 0], coordinates[:,1 ]] = color 

    return img

def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

def find_bounding_box(contour, img):

    box = []

    Xmin = numpy.min(contour[:,0])
    Xmax = numpy.max(contour[:,0])
    Ymin = numpy.min(contour[:,1])
    Ymax = numpy.max(contour[:,1])
    box.append([Xmin, Xmax, Ymin, Ymax])
    r = [box[0][0],box[0][1],box[0][1],box[0][0], box[0][0]]
    c = [box[0][3],box[0][3],box[0][2],box[0][2], box[0][3]]
    rr, cc = polygon_perimeter(r, c, img.shape)

    return [min(rr), min(cc)]

def get_maxpoints(contour):

    Xmin = numpy.min(contour[:,0])
    Xmax = numpy.max(contour[:,0])
    Ymin = numpy.min(contour[:,1])
    Ymax = numpy.max(contour[:,1])

    return numpy.array((Xmin,Ymin)), numpy.array((Xmax,Ymax)) 


def writeContour(letter,curClass, is_plate_one):
    

    img = color.rgb2gray(letter)
    image_8bit = numpy.uint8(img * 255)

    contours = measure.find_contours(img, 0.3)

    ll, ur = numpy.min(contours[1],0), numpy.max(contours[1],0)
    wh_max = ur - ll
    result = []
    for n, contour in enumerate(contours):
        ll, ur = numpy.min(contour,0), numpy.max(contour,0)
        wh = ur - ll
        if wh_max[0] - 2  < wh[0]:
            result.append(contour)
    curClass = [curClass[i:i+1] for i in range(0, len(curClass), 1)]
    curClass = ['carplate'] + curClass
    # result.sort(key=lambda x:find_bounding_box(x, img))
    # result = sorted(result, key=lambda ctr: find_bounding_box(ctr, img)[0] + find_bounding_box(ctr, img)[1] * img.shape[1] )
    # cntsSorted = sorted(result, key=lambda x:find_bounding_box(x, img))
    if (len(curClass) != len(result)):
        return str(0)
    boundingBoxes = [find_bounding_box(c, img) for c in result]
   

    ## ----------
    ##  key=lambda b:b[1][1], reverse=False for the ONE line sort 
    ##  key=lambda b:b[1][0], reverse=True for the TWO line sort
    # is_plate_one = True

    # fig, ax = plt.subplots()
    # ax.imshow(img, interpolation='nearest', cmap =plt.cm.gray)
    if is_plate_one:
        (result, boundingBoxes) = zip(*sorted(zip(result, boundingBoxes),
            key=lambda b:b[1][1], reverse=False))
        
        curClass.pop(0)
        result = result[1:len(result)]

    else:
        (result, boundingBoxes) = zip(*sorted(zip(result, boundingBoxes),
            key=lambda b:b[1][0], reverse=True))
        
        
        top_left, bottom_fight = get_maxpoints(result[len(result)-1])
        top_left_first, bottom_fight_first = get_maxpoints(result[0])
        dist_top = numpy.linalg.norm(top_left_first - top_left)
        dist_bottom = numpy.linalg.norm(bottom_fight_first - bottom_fight)
        if (dist_bottom > dist_top):
            curClass = curClass[3:] + curClass[1:3] + curClass[:1]
        else:
            curClass = list(reversed(curClass))
        buff = {}
        for box in boundingBoxes:
                if str(box[0]) in buff:
                    buff[str(box[0])] += 1
                else:
                    buff[str(box[0])] = 1
        for b in buff.values():
            if b >= 2:
                return str(0)
        # for n, contour in enumerate(result):
        #     plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        #     plt.show()

        curClass.pop()
        result = result[:len(result) - 1]

    
    
    json_txt = []
    
    

    for n, edge in enumerate(result):
        all_points_x = []
        all_points_y = []
        for i in range(len(edge)):
            all_points_x.append(int(edge[i,1]))
            all_points_y.append(int(edge[i,0]))
        
        json_txt.append({'shape_attributes': {'name': 'polyline', 'all_points_x':all_points_x, 
                                                        'all_points_y':all_points_y}, 'region_attributes':{'name': curClass[n], 'type': curClass[n]}  })
        
    return json_txt


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