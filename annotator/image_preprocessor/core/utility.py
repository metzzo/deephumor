import cv2


def to_jpeg(img):
    return cv2.imencode(".jpeg", img)[1].tostring()


def show_img_and_wait(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)