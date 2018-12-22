import cv2
import numpy as np

def nothing(x):
    pass

# Create a black image, a window
img = np.zeros((300,512,3), np.uint8)
img = cv2.resize(img, (0, 0), None, .25, .25)

grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# Make the grey scale image have three channels
# grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

cv2.namedWindow('image')
cv2.namedWindow('grey')

# create trackbars for color change
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)

while(1):
    cv2.imshow('image',img)
    cv2.imshow('grey',grey)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    s = cv2.getTrackbarPos(switch,'image')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


cv2.destroyAllWindows()