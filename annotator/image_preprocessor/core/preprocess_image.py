import numpy as np
import cv2
import pytesseract
from PIL import Image

from image_preprocessor.core.utility import show_img_and_wait


def preprocess_image(path):
    """

    Preprocesses the image given by path. Extracts the cartoon and a OCR version of the punchline

    :param path:
    :return: (cartoon, punchline)

    cartoon ... OpenCV image containing the cartoon
    punchline ... Punchline of the cartoon

    """
    original_img = cv2.imread(path, 0)
    real_original_img = original_img
    original_img_height, original_img_width = original_img.shape

    # Find Rectangle containing the cartoon
    # do this by finding the biggest contour which is the most rectangle like
    blurred = cv2.GaussianBlur(original_img, (7, 7), 0)
    edges = cv2.Canny(blurred, 100, 200, 3)
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_contour = None

    all_bounding_left = original_img_width
    all_bounding_top = original_img_height
    all_bounding_bottom = 0
    all_bounding_right = 0
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        all_bounding_left = min(all_bounding_left, x)
        all_bounding_top = min(all_bounding_top, y)
        all_bounding_bottom = max(all_bounding_bottom, y + h)
        all_bounding_right = max(all_bounding_right, x + w)
        if area > max_area and abs(w*h - area) < area * 0.1:
            max_area = area
            max_contour = c
    if max_contour is None:
        return original_img, 'unknown', real_original_img

    # cartoon should be on top => rotate if not
    x, y, w, h = cv2.boundingRect(max_contour)
    top = y - all_bounding_top
    left = x - all_bounding_left
    right = all_bounding_right - (x + w)
    bottom = all_bounding_bottom - (y + h)

    if right > top and right > left and right > bottom:
        # the text is on the right => +90°
        blurred = cv2.transpose(blurred)
        blurred = cv2.flip(blurred, flipCode=1)
        original_img = cv2.transpose(original_img)
        original_img = cv2.flip(original_img, flipCode=1)

    if left > top and left > right and left > bottom:
        # the text is on the left => -90°
        blurred = cv2.transpose(blurred)
        blurred = cv2.flip(blurred, flipCode=0)
        original_img = cv2.transpose(original_img)
        original_img = cv2.flip(original_img, flipCode=0)

    if top > bottom and top > left and top > right:
        # the text is on the bottom => +189°
        blurred = cv2.transpose(blurred)
        blurred = cv2.flip(blurred, flipCode=0)
        blurred = cv2.transpose(blurred)
        blurred = cv2.flip(blurred, flipCode=0)
        original_img = cv2.transpose(original_img)
        original_img = cv2.flip(original_img, flipCode=1)
        original_img = cv2.transpose(original_img)
        original_img = cv2.flip(original_img, flipCode=1)

    # show_img_and_wait(original_img)

    original_img_height, original_img_width = original_img.shape

    edges = cv2.Canny(blurred, 100, 200, 3)
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_contour = None

    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if area > max_area and abs(w * h - area) < area * 0.1:
            max_area = area
            max_contour = c
    if max_contour is None:
        return original_img, 'unknown', real_original_img

    # cut out cartoon
    mask = np.zeros_like(blurred)  # create empty mask
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

    if max_x - min_x > 0 and max_y - min_y > 0:
        cv2.rectangle(edges, (min_x, min_y), (max_x, max_y), (255, 0, 0), 1)

        # apply tesseract
        punchline = punchline[min_y:max_y, min_x:max_x]
        punchline = cv2.resize(punchline, None, fx=8, fy=8)
        punchline = 255 - punchline  # invert color, so we can use the threshold function
        _, punchline = cv2.threshold(punchline, 45, 255, cv2.THRESH_TOZERO)  # remove background noise
        punchline = cv2.equalizeHist(punchline)  # normalize brightness
        _, punchline = cv2.threshold(punchline, 140, 255, cv2.THRESH_TOZERO)  # now make text "thinner"
        punchline = 255 - punchline  # back to normal
        _, punchline = cv2.threshold(punchline, 0, 255, cv2.THRESH_OTSU)
        punchline_text = pytesseract.image_to_string(Image.fromarray(punchline), lang='eng')
    else:
        punchline_text = 'unknown'

    return out, punchline_text, real_original_img
