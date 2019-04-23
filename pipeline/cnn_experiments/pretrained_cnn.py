import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms

from cnn_experiments.base_model import BaseCNNModel
from datamanagement.cartoon_dataset import CartoonCNNDataset
from evaluation.accuracy_evaluation import AccuracyEvaluation

import numpy as np
import cv2

def auto_canny(image, sigma=0.7):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def edge_detection_impl(x):
    x = np.array(x)
    x = x[..., ::-1]

    img = cv2.blur(x, (5, 5))
    newImg = np.zeros(img.shape, np.uint8)

    # TODO: instead of simple edge detection use stylized imagenet
    thresh = auto_canny(image=img)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(newImg, contours, -1, 255, 1)

    old_size = newImg.shape[:2]

    ratio = float(224) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    newImg = cv2.resize(newImg, (new_size[1], new_size[0]))

    delta_w = 224 - new_size[1]
    delta_h = 224 - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    newImg = cv2.copyMakeBorder(newImg, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    #cv2.imshow('swag', newImg)
    # cv2.imshow('swag2', newImg2)
    #cv2.waitKey(0)

    return Image.fromarray(newImg) #cv2.cvtColor(newImg, cv2.COLOR_GRAY2BGR))

class PretrainedCNNCartoonCNNModel(BaseCNNModel):
    class Network(nn.Module):
        def __init__(self):
            super(PretrainedCNNCartoonCNNModel.Network, self).__init__()

            self.model_conv = torchvision.models.resnet18()

            for param in self.model_conv.layer1.parameters():
                param.requires_grad = False
            for param in self.model_conv.layer2.parameters():
                param.requires_grad = False
            for param in self.model_conv.layer3.parameters():
                param.requires_grad = False
            for param in self.model_conv.layer4.parameters():
                param.requires_grad = False
            self.model_conv.fc = nn.Linear(512, 100)
            self.fc2 = nn.Linear(100, 8)

        def forward(self, x):
            x = F.relu(self.model_conv(x))
            x = self.fc2(x)
            return x

        def load_state_dict(self, state_dict, strict=True):
            return self.model_conv.load_state_dict(state_dict=state_dict['state_dict'], strict=strict)

    def get_predictions(self, outputs):
        return torch.max(outputs, 1)[1]

    @property
    def get_network_class(self):
        return PretrainedCNNCartoonCNNModel.Network

    @property
    def optimization_parameters(self):
        return self.network.model_conv.parameters()

    def get_train_transformation(self):
        return [
            transforms.Lambda(edge_detection_impl),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]

    def get_validation_transformation(self):
        return [
            transforms.Lambda(edge_detection_impl),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]

    def load_image(self, img_name):
        return super(PretrainedCNNCartoonCNNModel, self).load_image(img_name=img_name).convert('RGB')

    @property
    def Dataset(self):
        return CartoonCNNDataset

    def get_input_and_label(self, data):
        _, image, labels = data
        """labels = labels.double()
        s = torch.from_numpy(np.random.normal(0, 0.5, len(labels)))
        labels += s
        labels = labels.round().long()
        labels = torch.clamp(labels, 1, 7)"""

        return image, labels
    @property
    def train_evaluations(self):
        return super(PretrainedCNNCartoonCNNModel, self).train_evaluations + [AccuracyEvaluation]

    @property
    def validation_evaluations(self):
        return super(PretrainedCNNCartoonCNNModel, self).validation_evaluations + [AccuracyEvaluation]


