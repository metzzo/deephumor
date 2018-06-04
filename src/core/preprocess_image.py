import numpy as np
import cv2

from core.utility import show_img_and_wait


def preprocess_image(path):
    """

    Preprocesses the image given by path. Extracts the cartoon and a OCR version of the punchline

    :param path:
    :return: (cartoon, punchline)
    """
    original_img = cv2.imread(path, 0)

    # Find Rectangle containing the cartoon
    # do this by finding the biggest contour which is the most rectangle like
    blurred = cv2.GaussianBlur(original_img, (7, 7), 0)
    edges = cv2.Canny(blurred, 100, 200, 3)
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_contour = None

    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        print(area)
        if area > max_area and abs(w*h - area) < area * 0.1:
            max_area = area
            max_contour = c
    assert max_contour is not None

    # cut out cartoon
    mask = np.zeros_like(original_img)  # create empty mask
    cv2.drawContours(mask, [max_contour], 0, 255, -1)  # draw filled contour into mask
    kernel = np.ones((16, 16), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)  # remove borders from mask
    out = np.zeros_like(original_img)
    out[mask == 255] = original_img[mask == 255]  # copy images over where mask is 255
    (x, y) = np.where(mask == 255)  # crop out image
    (topx, topy) = (np.min(x) + 1, np.min(y) + 1)
    (bottomx, bottomy) = (np.max(x) - 1, np.max(y) - 1)
    out = out[topx:bottomx + 1, topy:bottomy + 1]

    show_img_and_wait(img=out)
    pass