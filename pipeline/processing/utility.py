import cv2
import matplotlib.pyplot as plt
import numpy as np


def auto_canny(image, sigma=0.7):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def edge_detection(x, to_pil=True):
    x = np.array(x)
    x = x[..., ::-1]

    img = cv2.blur(x, (5, 5))
    newImg = np.zeros(img.shape, np.uint8)

    thresh = auto_canny(image=img)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(newImg, contours, -1, 255, 3)

    cv2.imshow('swag', newImg)
    cv2.waitKey(0)

    return Image.fromarray(newImg) if to_pil else newImg

def take_spectogram(**kwargs):
    f = np.fft.fft2(kwargs['image'])
    fshift = np.fft.fftshift(f)
    kwargs['image'] = np.log(np.abs(fshift))
    return kwargs

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(npimg[0], cmap='gray')
    plt.show()


import torch
from PIL import Image, ImageOps
import numpy as np

import torchvision
import torchvision.transforms.functional as F


class Invert(object):
    """Inverts the color channels of an PIL Image
    while leaving intact the alpha channel.
    """

    def invert(self, img):
        r"""Invert the input PIL Image.
        Args:
            img (PIL Image): Image to be inverted.
        Returns:
            PIL Image: Inverted image.
        """
        if not F._is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        if img.mode == 'RGBA':
            r, g, b, a = img.split()
            rgb = Image.merge('RGB', (r, g, b))
            inv = ImageOps.invert(rgb)
            r, g, b = inv.split()
            inv = Image.merge('RGBA', (r, g, b, a))
        elif img.mode == 'LA':
            l, a = img.split()
            l = ImageOps.invert(l)
            inv = Image.merge('LA', (l, a))
        else:
            inv = ImageOps.invert(img)
        return inv

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be inverted.
        Returns:
            PIL Image: Inverted image.
        """
        return self.invert(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'
