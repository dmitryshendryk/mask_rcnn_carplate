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

import common
from skimage import measure,data,color
FONT_DIR = "./fonts"
FONT_HEIGHT = 25  # Pixel size to which the chars are resized
FONT_TYPE=0
OUTPUT_SHAPE = (40, 100)

CHARS = common.CHARS + " "

#use tensorflow to augmentate the pic
import tensorflow as tf
def tensor_augmentation(image):
    if numpy.mean(image)<160/255 :
        return image
    #print (image.shape)
    #image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    img_tensor = tf.convert_to_tensor(image)

    output_img=image
    with tf.Session() as sess:
        br=random.uniform(0,0.45)
        after_ag=tf.image.adjust_brightness(img_tensor,br)
        #con=random.uniform(-0.3,0.3)

        #after_ag=tf.image.adjust_contrast(after_ag,con)
        output_img=after_ag.eval(session=sess)
    sess.close()
    #del sess
    #output_img=cv2.cvtColor(output_img,cv2.COLOR_RGB2GRAY)
    return  output_img

def blur(img,type=0):
    if type==1:
        size=random.choice([3,5])
    elif type==2:
        size=random.choice([5,7,9])
    elif type==3:
        size = random.choice([7, 9,11])
    else:
        size = random.choice([3,5,7])
    img_blur = cv2.GaussianBlur(img,(size,size),0)
    # img=img.astype(numpy.float32)
    # img_blur = cv2.bilateralFilter(src=img, d=random.randint(1,7), sigmaColor=random.randint(4,15), sigmaSpace=random.randint(1,15))
    return img_blur
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
def rand_colors():
    i=random.randint(1,2)
    plate_color=0
    text_color=0
    if i%2==0:
        #yellow
        plate_color=random.randint(85,130)
    else:
        plate_color = random.randint(170,235)
    text_color=random.randint(10,40)
    return text_color/255.0,plate_color/255.0

def pick_colors():
    first = True
    while first or plate_color - text_color < 0.3:
        text_color = random.random()
        plate_color = random.random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    #print (text_color,plate_color)
    return text_color, plate_color


def make_affine_transform(from_shape, to_shape,
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=0.1,type=0):
    out_of_bounds = False

    ##two small and big rotation are not allowed to come up in the same time
    #print(FONT_HEIGHT)
    if FONT_HEIGHT in range(17,19):
        rotation_variation*=0.1
    elif FONT_HEIGHT in range(19,21):
        rotation_variation*=0.3
    elif FONT_HEIGHT in range(21,23):
        rotation_variation*=0.5
    elif FONT_HEIGHT in range(23,26):
         rotation_variation*=0.8

    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T
    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    roll = random.uniform(-0.6, 0.6) * rotation_variation * 0.5
    if type==1: #single
        pitch = random.uniform(-1*60*numpy.pi/180, 60*numpy.pi/180)
        yaw =  random.uniform(-1*30*numpy.pi/180, 30*numpy.pi/180)
    else:
        pitch = random.uniform(-1 * 30 * numpy.pi / 180, 30 * numpy.pi / 180)
        yaw = random.uniform(-1 * 60 * numpy.pi / 180, 60 * numpy.pi / 180)
    pitch*=rotation_variation
    yaw*=rotation_variation
    #print(roll,yaw,pitch)
    # roll=0
    # pitch=-0.8
    # yaw=0
    #print(roll,pitch,yaw)

    # Compute a bounding box on the skewed input image (`from_shape`).
    #print("all_angle", roll * 180 / numpy.pi, pitch * 180 / numpy.pi, yaw * 180 / numpy.pi)
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
    #print(M)
    return M, out_of_bounds,center_to,center_from,pitch,yaw


def generate_code():
    #ormat={"{}{}{}{} {}{}{}","{}{}{}{}{}{}{}"}
    input = []
    all_letters=0
    all_digitis=0
    #print(FONT_TYPE)

    all_letters=common.LETTERS
    all_digitis=common.DIGITS


    for i in range(7):
            if random.randint(0,1)==0:
                input.append( random.choice(all_letters))
            # elif random.randint(0,2)==1:
            #     input.append(random.choice(common.DIGITS))
            else:
                input.append(random.choice( all_digitis))
    space_pos=random.randint(1,5)
    code=""
    for i in range(7):
        if i !=space_pos:
            code+=input[0]
            input=input[1:]
            #print(input)
        else:
            code+=" "
    #return "{}{} {}{}{}{}".format(input[0],input[1],input[2],input[3],input[4],input[5],input[6])
    return  code

    # return "{}{}{}{} {}{}{}".format(
    #     random.choice(common.LETTERS),
    #     random.choice(common.LETTERS),
    #     random.choice(common.DIGITS),
    #     random.choice(common.DIGITS),
    #     random.choice(common.LETTERS),
    #     random.choice(common.LETTERS),
    #     random.choice(common.LETTERS))

def generate_code_two_line():
    all_letters = 0
    all_digitis = 0
    #print(FONT_TYPE)

    all_letters = common.LETTERS
    all_digitis = common.DIGITS




    firstline= " {}{} ".format( random.choice(all_letters), random.choice(all_letters))
    secondline="{}{}{}{}".format( random.choice(all_digitis), random.choice(all_digitis), random.choice(all_digitis), random.choice(all_digitis))

    return firstline,secondline

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
#####this function is aim to gengerate random contrast and try to make the boundary of the character broken which happend in real world
def random_contrast(img):
    min=numpy.min(img)
    max=numpy.max(img)
    # cv2.namedWindow("temp")
    # cv2.imshow("temp",img)
    # cv2.waitKey(0)
    average=(min+max)/2
   # print(average)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            point=img[row][col]
            if point>average: #palte part
                e=random.uniform(0.0,0.002)
                if random.randint(0,1):
                    img[row][col]+=e
                else:
                    img[row][col]-=e
            elif point<average:
                #deternmine wheter this is boundary or not
                e=random.uniform(0,0.005)
                flag=True
                for i in range(-2,2):#there is no chance that it can go outside of the array
                    for j in range(-2,2):
                        if row+i<img.shape[0] and col+j<img.shape[1] and img[row+i][col+j]>average:#near the bound
                            img[row][col]+=e*2.0
                            flag=False
                            break
                if not flag:
                    img[row][col]+=e





def generate_two_lines_plate(font_height,char_ims):
    h_padding = 0.5 * font_height
    v_padding = random.uniform(0.02,0.1) * font_height
    med=random.uniform(0.0,1.0)*v_padding
    if not FONT_TYPE:
     er=random.uniform(0.001,0.05)
    else:
     er = random.uniform(0.00005, 0.0003)
    first_code_line,second_code_line = generate_code_two_line()

    all = second_code_line + first_code_line
    #don't use average width    use min
    average_width = sum(char_ims[c].shape[1] for c in second_code_line) / (len(second_code_line))

    width_list=[]
    for c in all:
        width_list.append(char_ims[c].shape[1])

    width_list.sort()
    #print(width_list)
    if '1' in all:
        min_width=width_list[1]
    else:
        min_width=width_list[0]
    spacing = min_width *er

    h_padding+=font_height * 0.01*(0.01-er)

    radius = 1 + int(font_height * 0.2 * random.random())




    max = 0
    for c in all:

        char_im=char_ims[c]
        if char_im.shape[1] > max:
            max = char_im.shape[1]

    text_width = 4*max
    text_width += (len(second_code_line) - 1) * spacing

    #find max and use it


    #print(first_line_width,text_width)

    out_shape = (int(font_height*2 + v_padding * 2+med),
                 int(text_width + h_padding * 2))
    center=[out_shape[0]/2,out_shape[1]/2]

    text_color, plate_color = rand_colors()

    text_mask = numpy.zeros(out_shape)

    x = h_padding
    y = v_padding
    count=0
    codes=[]
    imgs=[]
    for c in first_code_line:
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        start=int(max/2-char_im.shape[1]/2)
        end=int(max/2+char_im.shape[1]/2)
        text_mask[iy:iy + char_im.shape[0], ix+start:ix+end] = char_im

        if c != ' ':
            codes.append(c)

            imgs.append([[iy,ix+start],[iy + char_im.shape[0]-1,ix+start],[ iy + char_im.shape[0]-1,ix+end-1],[iy, ix + end - 1]])
        x += max + spacing

        #count+=1

    x = h_padding
    y = v_padding
    for d in second_code_line:
        char_im = char_ims[d]

        ix, iy = int(x), int(y+med+font_height)
        start = int(max / 2 - char_im.shape[1] / 2)
        end = int(max / 2 + char_im.shape[1] / 2)

        if d != ' ':
            codes.append(d)
            imgs.append(
                [[iy, ix + start], [iy + char_im.shape[0] - 1, ix + start], [iy + char_im.shape[0] - 1, ix + end - 1],
                 [iy, ix + end - 1]])
        text_mask[iy:iy + char_im.shape[0], ix+start:ix+end] = char_im

        x += max + spacing


    plate = (numpy.ones(out_shape) * plate_color * (1. - text_mask) +
             numpy.ones(out_shape) * text_color * text_mask)

    x=int(random.uniform(1.0/6.0,1.0/5.0)*plate.shape[0])
    y=int(random.uniform(1.0/10.0,1.0/11.0)*plate.shape[1])

    c_color=random.uniform(-0.1,0.1)
    cv2.circle(plate,(x,y),2,c_color,-1)
    cv2.circle(plate,(plate.shape[1]-x,y),2,c_color,-1)

    ratios=(OUTPUT_SHAPE[1]-60)/100
    maxium=int(3+27*ratios)
    min=int(1+6*ratios)

    h=random.randint(min,int(maxium*2/3))
    w=random.randint(min,maxium)
    start_c=random.uniform(0.01,0.05)
    for i in range(h):
        for j in range(plate.shape[1]):
            plate[i][j]-=(start_c-random.uniform(0.2,1.0)*i/200.0)
            plate[plate.shape[0]-1-i][j]-=(start_c- random.uniform(0.2,1.0)*i/200.0)
    for i in range(w):
        for j in range(plate.shape[0]):
            plate[j][i]-=(start_c-random.uniform(0.2,1.0)*i/200.0)
            plate[j][plate.shape[1]-1-i]-=(start_c- random.uniform(0.2,1.0)*i/200.0)


    code=first_code_line+second_code_line

    #plate = blur(plate)
    plate[:, 0:random.randint(1,10)] -= random.uniform(0.03, 0.06)
    plate[0:random.randint(1,5), :] -= random.uniform(0.03, 0.06)

    #print(len(imgs))
    return plate, rounded_rect(out_shape, radius), code.replace(" ", ""),codes,imgs,center#set the name
def generate_plate(font_height, char_ims):
    h_padding = random.uniform(0.2, 0.4) * font_height
    v_padding = random.uniform(0.1, 0.3) * font_height
    code = generate_code()
    if not FONT_TYPE:
        spacing = font_height * random.uniform(0, 0.05)
    else:
        width_list = []
        for c in code:
            width_list.append(char_ims[c].shape[1])

        width_list.sort()
        # print(width_list)
        if '1' in code:
            spacing = width_list[1] * random.uniform(0.1, 0.4)
        else:
            spacing = width_list[0] * random.uniform(0.1, 0.4)

    radius = 1 + int(font_height * 0.1 * random.random())




    percent = font_height / char_ims[code[0]].shape[0]


    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2),
                 int(text_width + h_padding * 2))
    center = [out_shape[0] / 2, out_shape[1] / 2]
    #text_color should be around [
    text_color, plate_color = rand_colors()

    text_mask = numpy.zeros(out_shape)

    x = h_padding
    y = v_padding
    codes = []
    imgs = []
    for c in code:

        empty_background=numpy.zeros(out_shape)
        char_im = char_ims[c]


        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        if c != ' ':
            #back_up
            #empty_background[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = 1
            #back_up_end
            # empty_background[iy][ix]=1
            # empty_background[iy+char_im.shape[0]][ix]=1
            #
            # empty_background[iy+char_im.shape[0]][ix+char_im.shape[1]]=1
            # empty_background[iy][ix + char_im.shape[1]] = 1
            codes.append(c)

            imgs.append([[iy,ix],[iy+char_im.shape[0],ix],[iy+char_im.shape[0],ix+char_im.shape[1]],[iy,ix + char_im.shape[1]]])

        x += char_im.shape[1] + spacing

    plate = (numpy.ones(out_shape) * plate_color * (1. - text_mask) +
             numpy.ones(out_shape) * text_color * text_mask)

    h = random.randint(1, 3)
    w = random.randint(5, 15)
    start_c = random.uniform(0.01, 0.03)
    for i in range(h):
        for j in range(plate.shape[1]):
            plate[i][j] -= (start_c - random.uniform(0.5,2.0)*i / 255.0)
            plate[plate.shape[0] - 1 - i][j] -= (start_c - random.uniform(0.5,2.0)*i / 125.0)
    for i in range(w):
        for j in range(plate.shape[0]):
            plate[j][i] -= (start_c - random.uniform(0.5,2.0)*i / 255.0)
            plate[j][plate.shape[1] - 1 - i] -= (start_c - random.uniform(0.5,2.0)*i / 255.0)
    #plate = blur(plate)
    # if not "1" in code:
    #     plate = blur(plate)
    #plate = blur(plate)
    #加深边界
    plate[:,0:random.randint(1,5)]-=random.uniform(0.01, 0.06)
    plate[0:random.randint(1,5),:]-=random.uniform(0.01, 0.06)

    #print(codes)
    return plate, rounded_rect(out_shape, radius), code.replace(" ", ""),codes,imgs,center


def generate_bg(num_bg_images):
    found = False
    length = len(num_bg_images)
    while not found:
        index = int(random.randint(0, length - 1))
        img_name = num_bg_images[index]
        fname = "./bgs/"+img_name
        # bg = cv2.imread(fname)
        bg = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) / 255.
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True

    # x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    # y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    # bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]
    #bg=blur(bg,2)
    bg = cv2.resize(bg,(OUTPUT_SHAPE[1],OUTPUT_SHAPE[0]))

    return bg,img_name


def generate_im(char_ims, num_bg_images):

    #the possibility of one line or two line is now 4:6

    plate_mask=0
    code=0
    ran=[0,0,0,1,1]
    c_code=0
    idv_img=0
    if random.randint(1,5)>=2:
          plate, plate_mask, code,c_code,idv_img = generate_plate(FONT_HEIGHT, char_ims)
    else:
           plate, plate_mask, code,c_code,idv_img = generate_two_lines_plate(FONT_HEIGHT, char_ims)
    bg, img_name = generate_bg(num_bg_images)
    M, out_of_bounds = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            min_scale=1.5,
                            max_scale=2,
                            rotation_variation=0.8,
                            scale_variation=1.5,
                            translation_variation=0.7)

    cv2.imshow("plate maske",plate_mask)
    cv2.waitKey(0)
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    # cv2.imwrite('mask/'+img_name, plate_mask* 255.)
    # cv2.imwrite('plate.jpg', plate)


    out = plate * plate_mask + bg * (1.0 - plate_mask)
    # cv2.imwrite('plate.jpg', plate)
    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))
    noise_sc=random.uniform(0.3,0.8)
    out += numpy.random.normal(scale=noise_sc, size=out.shape)
    out = numpy.clip(out, 0., 1.)
    # cv2.imwrite('plate.jpg', out* 255.)
    return plate_mask,out, code, not out_of_bounds

#This is aim to recongise the car plate and contend on it together
#so far is only for test use
def random_padding(contours,img_size):
    height=img_size[0]
    width=img_size[1]
    #print(height,width)
    new_contours=[]
    for edg in contours:
        #print(edg)
        if random.randint(0,1) == 1:
            continue
        r = random.choice([-1,1, 2])

        left_up=[edg[0]-r,edg[1]-r]
        right_up=[edg[0]+r,edg[1]-r]
        right_down=[edg[0]+r,edg[1]+r]
        left_down=[edg[0]-r,edg[1]+r]

        new_edge=[left_up,right_up,right_down,left_down]

        for p in new_edge:

            p[0]=p[0] if p[0]>0 else 0
            p[0] = p[0] if p[0] < width else width-1
            p[1]=p[1] if p[1]>0 else 0
            p[1]=p[1] if p[1]<height else height-1
        new_contours.append(new_edge)
    return  new_contours
        #for point in edg:




def multi_generate_im(char_ims,num_bg_images):
    bg, img_name = generate_bg(num_bg_images)
    code = 0
    ran = [0, 0, 0, 1, 1]
    c_code=0
    idv_img=0
    center=0
    tg=0
    if random.randint(1, 5) >= 2 and FONT_HEIGHT>22:
        plate, plate_mask,code, c_code,idv_img,center = generate_plate(FONT_HEIGHT, char_ims)
        plate=tensor_augmentation(plate)

    else:
        plate, plate_mask,code, c_code,idv_img,center = generate_two_lines_plate(FONT_HEIGHT, char_ims)
        plate = tensor_augmentation(plate)
        tg=1
    #random_contrast(plate)
    M, out_of_bounds,c_t,c_f,pitch,yaw = make_affine_transform(
        from_shape=plate.shape,
        to_shape=bg.shape,
        min_scale=0.95,
        max_scale=1.0,
        rotation_variation=1.0,
        scale_variation=1.5,
        translation_variation=1.0,type=tg)

    plate = cv2.warpAffine(plate, M,  (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))
    labels=[]
    contours=[]
    M_aug = numpy.zeros((M.shape[0]+1,M.shape[1]))
    M_aug[:2,:]=M
    M_aug[-1,-1]=1


    for i in range(len(c_code)):

        labels.append(c_code[i])
        edge = []
        cur_img=idv_img[i]

        #determine if need padding
        r=0
        if random.randint(0, 1) == 1:
           r=random.choice([-1,1])

        count=0
        #print(plate.shape)
        #print(cur_img)


        for point in cur_img:
           point=[point[1],point[0]]
           after_trans=numpy.dot(M_aug,numpy.hstack((numpy.array(point),1)).reshape(-1,1))
           re_p=after_trans
           y=re_p[0]
           x=re_p[1]
           edge.append([x,y])
           count+=1

        contours.append(numpy.array(edge))
        # cv2.imshow(" ", idv_img[i])
        # cv2.waitKey(0)
    #print(contours)
    # cv2.imwrite('mask/'+img_name, plate_mask* 255.)
    # cv2.imwrite('plate.jpg', plate)
    #contours=random_padding(contours,plate.shape)
    out = plate * plate_mask + bg * (1.0 - plate_mask)

    # cv2.imwrite('plate.jpg', plate)
    if OUTPUT_SHAPE[1]>135 and abs(pitch)<25 * numpy.pi / 180 and abs(yaw)<25*numpy.pi/180:
        out = blur(out,3)
    if OUTPUT_SHAPE[1]>110 and abs(pitch)<40 * numpy.pi / 180 and abs(yaw)<40*numpy.pi/180:
        out = blur(out,2)
    elif OUTPUT_SHAPE[1]>75  and abs(pitch)<45 * numpy.pi / 180 and abs(yaw)<45*numpy.pi/180:
        out=blur(out,0)
    else:
        out=blur(out,1)
    #out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    noise_sc = random.uniform(0.03, 0.08)
    out += numpy.random.normal(scale=noise_sc, size=out.shape)
    out = numpy.clip(out, 0., 1.)

    meanpixel=numpy.mean(out)
    #print(meanpixel)
    # cv2.imwrite('plate.jpg', out* 255.)
    #contours = measure.find_contours(img, 0.95)
    #if len(contours) < 1:
    #there should be n contours

    #edge = contours[0]
    contend_map={}

    #for contour in contours:

    return  out, labels,contours,meanpixel

def load_fonts(folder_path):
    font_char_ims = {}
    fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]
    #print("font height  ",FONT_HEIGHT)
    for font in fonts:
        font_char_ims[font] = dict(make_char_ims(os.path.join(folder_path,
                                                              font),
                                                 FONT_HEIGHT))

    #print(FONT_HEIGHT)
    # fonttype='TT0756M_.ttf'
    # for c in common.H_CHARS:
    #
    #
    #      font_char_ims[fonttype][c]=font_char_ims[fonttype][c][int(font_char_ims[fonttype][c].shape[0]*1/4):-1,:]
    #
    #      scale=FONT_HEIGHT/font_char_ims[fonttype][c].shape[0]
    #      font_char_ims[fonttype][c]=cv2.resize(font_char_ims[fonttype][c],(int(font_char_ims[fonttype][c].shape[1]),FONT_HEIGHT),cv2.INTER_LINEAR)
    #
    #      #font_char_ims[fonttype][c]=cv2.resize(font_char_ims[fonttype][c],(font_char_ims[fonttype][c].shape[0],))
    #      cv2.imshow(" ", font_char_ims[fonttype][c])
    #      cv2.waitKey(0)
    # 是否采用随机亮度变化









    return fonts, font_char_ims


def generate_ims():
    """
    Generate number plate images.
    :return:
        Iterable of number plate images.
    """
    variation = 1.0

    # fonts, font_char_ims = load_fonts(FONT_DIR)
    # for c in common.CHARS :
    #     font_char_ims["hussarbd-web.ttf"][c]=font_char_ims["hussarbd-web.ttf"][c][int(font_char_ims["hussarbd-web.ttf"][c].shape[0]*1/4+2):-1,:]
    # t_u=[0,0]
    # t_d=[0,0]
    # for c in common.CHARS:
    #     height=font_char_ims["hussarbd-web.ttf"][c].shape[0]
    #     width=font_char_ims["hussarbd-web.ttf"][c].shape[1]
    #     font_char_ims["hussarbd-web.ttf"][c]=font_char_ims["hussarbd-web.ttf"][c][int(height*1/4):-1,:]
    #     print(height)
    #     t_d[0]+=height
    #     t_d[1]+=width
    # for c in common.CHARS:
    #     height=font_char_ims["UKNumberPlate.ttf"][c].shape[0]
    #     width=font_char_ims["UKNumberPlate.ttf"][c].shape[1]
    #     print(height)
    #     t_u[0] += height
    #     t_u[1] += width
    # av_u=(t_u[0]/33.0,t_u[1]/33.0)
    # av_d=(t_d[0]/33.0,t_d[1]/33.0)
    # print(av_u,av_d)
    #return

    #mataince a array to save the ratio of ukfont
    ratios={}
    fonts, font_char_ims = load_fonts(FONT_DIR)
    for c in common.CHARS:
        ratios[c]=font_char_ims["UKNumberPlate.ttf"][c].shape

    #print(ratios)
    #return
    num_bg_images = os.listdir("./bgs")
    while True:
        #font_height = random.randint(16, 30)
        output_width=random.randint(60,160)
        output_height=int(output_width/100*random.randint(40,70))
        global OUTPUT_SHAPE
        global FONT_HEIGHT
        global FONT_TYPE
        global CHARS
        OUTPUT_SHAPE=(output_height,output_width)

        #FONT_HEIGHT= int(output_height/100*14+18)+random.randint(-2,2)



        # if fonttype=="hussarbd-web.ttf":
        #     FONT_HEIGHT = int((output_height / 100 * 14 + 18)*0.64)
        #
        # else:
        FONT_HEIGHT = int((output_width-60) /100 * 14 + 19) + random.randint(-2, 2)
        fonts, font_char_ims = load_fonts(FONT_DIR)
        fonttype = random.choice(fonts)
        #print(fonttype)
        if fonttype!="UKNumberPlate.ttf":
            FONT_TYPE=1
            for c in common.H_CHARS:
             font_char_ims[fonttype][c]=font_char_ims[fonttype][c][int(font_char_ims[fonttype][c].shape[0]*1/4):-1,:]
             # if c!="1":
             #     new_height=FONT_HEIGHT
             #     new_width=int(FONT_HEIGHT*(ratios[c][1]/ratios[c][0]))
             # else:
             new_height=FONT_HEIGHT
             new_width=font_char_ims["UKNumberPlate.ttf"][c].shape[1]
             font_char_ims[fonttype][c]=cv2.resize(font_char_ims[fonttype][c],(new_width,new_height),cv2.INTER_LINEAR)
            for c in common.R_CHARS:
                font_char_ims[fonttype][c] = font_char_ims["UKNumberPlate.ttf"][c]

        else:
            FONT_TYPE=0
        yield multi_generate_im(font_char_ims[fonttype], num_bg_images)



def gen_label(labels,contours,file,pic_index):

    img_name=pic_index
    file.writelines(img_name)
    file.writelines(' %d'%(len(labels)))
    #print(labels)
    #temp=[]
    for i in range(len(labels)):
        #print(i)
        current_label=labels[i]
        file.writelines(" "+current_label)
        current_contour=contours[i]
        #temp.append(current_contour)
        #file.writelines(" %d "%(int(len(current_contour)/5)+1))#reduce amount consider saw
        pts_num = len(current_contour)
        # for i in range(len(current_contour)):
        #         if i%5==0:
        #             pts_num+=1
            #print('pts_num:', pts_num)
        file.writelines(" %d "%pts_num)
        t1 = 0
        t2 = 0
        for i in range(len(current_contour)):

                file.writelines(str(int(current_contour[i, 1])))
                file.writelines(' ')

        # print('t1:',t1)
        for i in range(len(current_contour)):

                file.writelines(str(int(current_contour[i, 0])))
                if i<len(current_contour)-1:
                    file.writelines(' ')

    #plt.show(temp)
    #cv2.waitKey(0)
    file.writelines('\n')
    print("finish "+img_name)


import matplotlib.pyplot as plt
def show_contours(img,contours):# show all contours

    #print(contours)
    for edge in contours:
        for i in range(len(edge)-1):
             cv2.line(img,(edge[i][1],edge[i][0]),(edge[i+1][1],edge[i+1][0]),0,1)
        cv2.line(img,(edge[len(edge)-1][1],edge[len(edge)-1][0]),(edge[0][1],edge[0][0]),0,1)
    cv2.imshow("",img)
    cv2.waitKey(0)
    # fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    # ax0, ax1 = axes.ravel()
    # ax0.imshow(img, plt.cm.gray)
    # ax0.set_title('original image')
    #
    # rows, cols = img.shape
    # ax1.axis([0, rows, cols, 0])
    #
    # for contour in contours:
    #     #print(contour)
    #     ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
    # ax1.axis('image')
    # ax1.set_title('contours')
    # plt.show()
    #cv2.imshow(" ",img)
    #cv2.waitKey(0)

#if not os.path.exists('mask_Bigscale_train'):
    #os.mkdir("mask_Bigscale_train")
#if not os.path.exists('mask_Bigscale_val'):
    #os.mkdir("mask_Bigscale_val")

ROOT_DIR = os.path.abspath("../")


data_path = os.path.join(ROOT_DIR, 'ccs_dataset_new')
if not os.path.exists(data_path):
    os.mkdir(data_path)

output_label_txt = os.path.join(ROOT_DIR, 'ccs_dataset_new/label.txt')  
print(output_label_txt)
file = open(output_label_txt,'w')
im_gen = itertools.islice(generate_ims(),30000)
mean=0
for img_idx, (out,labels, contours,meanpixel) in enumerate(im_gen):
    fname =  os.path.join(ROOT_DIR, "ccs_dataset_new/{}.png".format(img_idx) )  
    cv2.imwrite(fname, out * 255.)
    #show_contours(out,contours)
    gen_label(labels,contours,file,"{}.png".format(img_idx))
    mean=(mean*img_idx+meanpixel)/(img_idx+1)
    #cv2.imwrite(fname2, plate_mask * 255.)
file.close()
print("mean pixel!!!!!!!!!!!",mean)
