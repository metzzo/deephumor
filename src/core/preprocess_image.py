import numpy as np
import cv2
import pytesseract
from PIL import Image

from core.utility import show_img_and_wait


def preprocess_image(path, show_img=False):
    """

    Preprocesses the image given by path. Extracts the cartoon and a OCR version of the punchline

    :param path:
    :return: (cartoon, punchline)

    cartoon ... OpenCV image containing the cartoon
    punchline ... Punchline of the cartoon

    """
    original_img = cv2.imread(path, 0)
    original_img_height, original_img_width = original_img.shape

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

    # extract text => below cartoon
    x, y, w, h = cv2.boundingRect(max_contour)
    punchline = original_img[y + h:y + h + (original_img_height - (y + h)), 0:original_img_width]
    blurred = cv2.GaussianBlur(punchline, (3, 3), 0)
    edges = cv2.Canny(blurred, 100, 200, 3)
    im2, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    text_height, text_width = edges.shape
    min_x, min_y = text_width, text_height
    max_x = max_y = 0
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x + w, max_x)
        min_y, max_y = min(y, min_y), max(y + h, max_y)

    assert max_x - min_x > 0 and max_y - min_y > 0

    cv2.rectangle(edges, (min_x, min_y), (max_x, max_y), (255, 0, 0), 1)

    # apply tesseract
    punchline = punchline[min_y:max_y, min_x:max_x]
    #punchline = cv2.GaussianBlur(punchline, (1, 1), 0)
    punchline = cv2.resize(punchline, None, fx=8, fy=8)
    punchline = 255 - punchline  # invert color, so we can use the threshold function
    _, punchline = cv2.threshold(punchline, 45, 255, cv2.THRESH_TOZERO)  # remove background noise
    punchline = cv2.equalizeHist(punchline)  # normalize brightness
    _, punchline = cv2.threshold(punchline, 140, 255, cv2.THRESH_TOZERO)  # now make text "thinner"
    punchline = 255 - punchline  # back to normal
    #punchline = cv2.medianBlur(punchline, 3)
    #punchline = cv2.adaptiveThreshold(punchline, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #_, punchline = cv2.threshold(punchline, 0, 255, cv2.THRESH_OTSU)
    punchline_text = pytesseract.image_to_string(Image.fromarray(punchline), lang='eng')
    print("Extracted Punchline: " + punchline_text)

    if show_img:
        show_img_and_wait(out)
        show_img_and_wait(punchline)

    return out, punchline_text
