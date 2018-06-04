import cv2

def show_img_and_wait(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)