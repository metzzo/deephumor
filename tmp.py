import cv2
import numpy as np

def edge(img, size=1):
    img = cv2.blur(img, (5, 5))
    newImg = np.zeros(img.shape, np.uint8)
    # ret, thresh = cv2.threshold(img, 127, 255, 0)
    thresh = cv2.Canny(img, 100, 200)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(newImg, contours, -1, (0,0,255), size)
    return newImg

cimg1 = cv2.imread('export/objects/cartoon_8_object_37.jpg', 1)
cimg2 = edge(cimg1)
img = np.hstack([cimg1, cimg2])
cv2.imshow('swg', img)
cv2.waitKey(0)

img = cv2.imread('export/real_hai.jpg', 1)
img = cv2.resize(img,None,fx=0.5,fy=0.5)
img2 = edge(img, size=4)
img3 = edge(img2)
img = np.hstack([img, img2, img3])
cv2.imshow('swg', img)
cv2.waitKey(0)