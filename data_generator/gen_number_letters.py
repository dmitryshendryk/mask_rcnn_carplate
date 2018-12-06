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

import cv2
import numpy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import matplotlib.pyplot as plt
from skimage import measure,data,color

import common

FONT_DIR = "./fonts"
FONT_HEIGHT = 32  # Pixel size to which the chars are resized

OUTPUT_SHAPE = (256, 256)

CHARS = common.CHARS + " "


def make_char_ims(font_path, output_height):
    font_size = output_height * 4

    font = ImageFont.truetype(font_path, font_size)

    height = max(font.getsize(c)[1] for c in CHARS)

    for c in CHARS:
        width = font.getsize(c)[0]
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
        plate_color = random.random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color


def writeContour(letter,curClass):
    # a = cv2.imread(img_path + '/' + img_name)
    from skimage import measure, data, color
    # import cv2

    img = color.rgb2gray(letter)
    # if img_name =='model1444.jpg':
    #     a=0
    # img=color.rgb2gray(data.horse())

  
    
    contours = measure.find_contours(img, 0.5)
    if len(contours) < 1:
        return ""
    edge = contours[0]

    # write txt
    # print(img_name)
    # curClass = img_name[9]
    output_txt =curClass + ' '

    pts_num = 0
    for i in range(len(edge)):
        if i % 5 == 0:
            pts_num += 1
    print('pts_num:', pts_num)
    output_txt +=str(pts_num)
    output_txt +=' '
    t1 = 0
    t2 = 0
    for i in range(len(edge)):
        if i % 5 == 0:
            output_txt +=str(int(edge[i, 1]))
            output_txt +=' '
            t1 += 1
    # print('t1:',t1)
    for i in range(len(edge)):
        if i % 5 == 0:
            output_txt +=str(int(edge[i, 0]))
            output_txt +=' '
            t2 += 1

    return output_txt

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


def generate_code(character):
    return "{}".format(
        # random.choice(common.LETTERS),
        # random.choice(common.LETTERS),
        # random.choice(common.DIGITS),
        # random.choice(common.DIGITS),
        # random.choice(common.LETTERS),
        # random.choice(common.LETTERS),
        random.choice(character))


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


def generate_plate(font_height, char_ims, character):
    h_padding = random.uniform(0.2, 0.1) * font_height
    v_padding = random.uniform(0.1, 0.1) * font_height
    spacing = font_height * random.uniform(-0.05, 0.05)
    radius = 1 + int(font_height * 0.1 * random.random())

    code = generate_code(character)
    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding ),
                 int(text_width + h_padding ))

    text_color, plate_color = pick_colors()
    
    text_mask = numpy.zeros(out_shape)
    
    x = h_padding
    y = v_padding 
    for c in code:
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing

    plate = (numpy.ones(out_shape) * plate_color * (1. - text_mask) +
             numpy.ones(out_shape) * 0.01 * text_mask)
    if random.randint(0,1) == 0:
        plate = cv2.blur(plate,(5,8))
    return plate, rounded_rect(out_shape, radius), code.replace(" ", ""),text_mask


def generate_bg(num_bg_images):
    found = False
    while not found:
        fname = "bgs_1/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
        bg = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) / 255.
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

    return bg


def generate_im(char_ims, num_bg_images, character):
    bg = generate_bg(num_bg_images)

    plate, plate_mask, code, number = generate_plate(FONT_HEIGHT, char_ims, character)
    
    M, out_of_bounds = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            min_scale=0.05,
                            max_scale=0.17,
                            rotation_variation=0.3,
                            scale_variation=0.8,
                            translation_variation=1.0)
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    # return the mask plate
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))
    # return the number mask
    number = cv2.warpAffine(number, M, (bg.shape[1], bg.shape[0]))
    label_txt = " "
    s = writeContour(number, code)
    label_txt += s
    out = plate * plate_mask + bg * (1.0 - plate_mask)

    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    out += numpy.random.normal(scale=0.05, size=out.shape)
    out = numpy.clip(out, 0., 1.)
   
    return number, out, code, not out_of_bounds, label_txt


def load_fonts(folder_path):
    font_char_ims = {}
    fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]
    for font in fonts:
        font_char_ims[font] = dict(make_char_ims(os.path.join(folder_path,
                                                              font),
                                                 FONT_HEIGHT))
    return fonts, font_char_ims


def generate_ims(character):
    """
    Generate number plate images.

    :return:
        Iterable of number plate images.

    """
    variation = 1.0
    fonts, font_char_ims = load_fonts(FONT_DIR)
    num_bg_images = len(os.listdir("bgs_1"))
    while True:
        yield generate_im(font_char_ims[random.choice(fonts)], num_bg_images, character)


def gen_number_txt(txt_file_train = 'label_data_train_letters.txt', img_source='mask_number_plate'):

    dirname, filename = os.path.split(os.path.abspath(__file__))
    #output_label_txt = '/home/jianfenghuang/Myproject/CarPlateRecog/deep-anpr-master/label_data_train.txt'
    output_label_txt = os.path.join(dirname, 'car_plate_data/' + txt_file_train)
    f_output = open(output_label_txt,'a')
    img_path = os.path.join(dirname, 'car_plate_data/' + img_source) 
    imgs = os.listdir(img_path)
    for img_name in imgs:

        a=cv2.imread(img_path+'/'+img_name)
        img=color.rgb2gray(a)
        # if img_name =='model1444.jpg':
        #     a=0
        # img=color.rgb2gray(data.horse())

        #detect all contours
        contours = measure.find_contours(img, 0.9)
        if len(contours)<1:
            continue
        edge = contours[0]

        # write txt
        print(img_name)
        f_output.writelines(img_name)
        f_output.writelines(' 1')
        curClass=img_name[9]
        f_output.writelines(' '+curClass+' ')

        pts_num=0
        for i in range(len(edge)):
            if i%5==0:
                pts_num+=1
        print('pts_num:', pts_num)
        f_output.writelines(str(pts_num))
        f_output.writelines(' ')
        t1=0
        t2=0
        for i in range(len(edge)):
            if i%5==0:
                f_output.writelines(str(int(edge[i,1])))
                f_output.writelines(' ')
                t1+=1
        # print('t1:',t1)
        for i in range(len(edge)):
            if i % 5 == 0:
                f_output.writelines(str(int(edge[i,0])))
                f_output.writelines(' ')
                t2 += 1
        # print('t2:', t2)
        f_output.writelines('\n')
        #  # draw contour
        # fig, axes = plt.subplots(1,2,figsize=(8,8))
        # ax0, ax1= axes.ravel()
        # ax0.imshow(img,plt.cm.gray)
        # ax0.set_title('original image')
        
        # rows,cols=img.shape
        # ax1.axis([0,rows,cols,0])
        # for n, contour in enumerate(contours):
        #     if n<1:
        #         print(contour)
        #         ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
        # ax1.axis('image')
        # ax1.set_title('contours')
        # plt.show()
    f_output.close()
    



def run(train_number, val_number, folder_train='train', folder_test='test_number_plate', txt_file_train = 'general_train.txt', txt_file_test='label_data_test_letters.txt'):
    
    labelPath1 = 'car_plate_data/' + txt_file_train
    f_labeltxt = open(labelPath1,'a')
    labelPath2 = 'car_plate_data/' + txt_file_test
    f_labeltxt2 = open(labelPath2,'a')


    if not os.path.exists("car_plate_data/" + folder_train):
        os.mkdir("car_plate_data/" + folder_train)

    if not os.path.exists("car_plate_data/" + folder_test):
        os.mkdir("car_plate_data/" + folder_test)

    if not os.path.exists('car_plate_data/mask_number_plate'):
        os.mkdir("car_plate_data/mask_number_plate")
    
    if not os.path.exists('car_plate_data/mask_number_plate_test'):
        os.mkdir("car_plate_data/mask_number_plate_test")
    
    dataset = ['0','1','2','3','4','5','6','7','8','9','Q','W','E','R','T','Y','U','I','O',
                       'P','A','S','D','F','G','H','J','K','L',
                       'Z','X','C','V','B','N','M' ] 
    test_dataset = ['0','1','2']
    for character in dataset:
        print("Start epoche for {}".format(character))
        im_gen = itertools.islice(generate_ims(character), train_number)
        for img_idx, (plate_mask, im, c, p, label_txt) in enumerate(im_gen):
        
            fname = "car_plate_data/"+ folder_train +"/{:08d}_{}_{}.png".format(img_idx, c,
                                                "1" if p else "0")
            img_name = "{:08d}_{}_{}.png".format(img_idx, c,
                                             "1" if p else "0")

            fname2 = "car_plate_data/mask_number_plate/{:08d}_{}_{}.png".format(img_idx, c,
                                                "1" if p else "0")
                                        
            # print(fname)
            print(str(img_idx) + "/" + str(train_number) + "-------" +fname)
            cv2.imwrite(fname, im * 255.)
            f_labeltxt.write(img_name + ' 1' + label_txt + "\n")
            cv2.imwrite(fname2, plate_mask * 255.)

        print("Finished epoche for {}".format(character))   


    for character in dataset:
        print("Start epoche for {}".format(character))
        im_gen = itertools.islice(generate_ims(character), val_number)
        for img_idx, (plate_mask, im, c, p, label_txt) in enumerate(im_gen):
        
            fname = "car_plate_data/"+ folder_test +"/{:08d}_{}_{}.png".format(img_idx, c,
                                                "1" if p else "0")
            img_name = "{:08d}_{}_{}.png".format(img_idx, c,
                                             "1" if p else "0")

            fname2 = "car_plate_data/mask_number_plate_test/{:08d}_{}_{}.png".format(img_idx, c,
                                                "1" if p else "0")
                                        
            # print(fname)
            print(str(img_idx)+ "/" + str(val_number) + "-------"+fname)
            cv2.imwrite(fname, im * 255.)
            f_labeltxt2.write(img_name + ' 1' + label_txt + "\n")
            cv2.imwrite(fname2, plate_mask * 255.)

        print("Finished epoche for {}".format(character))  