import torch

import torch.nn as nn
from torchvision import transforms

from architectures.base_model import BaseModel
from datamanagement.tuberlin_dataset import TUBerlinDataset
from evaluation.accuracy_evaluation import AccuracyEvaluation
from processing.utility import Invert


class TUBerlinClassificationModel(BaseModel):
    class Network(nn.Module):
        def __init__(self):
            super(TUBerlinClassificationModel.Network, self).__init__()

            self.seq = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=15, stride=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, 2),

                nn.Conv2d(64, 128, kernel_size=5, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(3, 2),

                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(3, 2),

                nn.Conv2d(256, 512, kernel_size=7, stride=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Dropout(p=0.5),

                nn.Conv2d(512, 512, kernel_size=1, stride=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Dropout(p=0.5),

                nn.Conv2d(512, 250, kernel_size=1, stride=1)
            )

        def forward(self, x):
            x = self.seq(x)
            x = x.view(x.size(0), -1)
            return x

    @property
    def get_network_class(self):
        return TUBerlinClassificationModel.Network

    def get_predictions(self, outputs):
        return torch.max(outputs, 1)[1]

    @property
    def optimization_parameters(self):
        return self.network.parameters()

    def get_train_transformation(self):
        return [
            transforms.RandomResizedCrop(225, scale=(0.9, 1.1), ratio=(0.75, 1.3333333333333333), interpolation=2),
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(),
            Invert(),
            transforms.ToTensor(),
        ]

    def get_validation_transformation(self):
        return [
            transforms.Resize(225),
            Invert(),
            transforms.ToTensor(),
        ]

    @property
    def Dataset(self):
        return TUBerlinDataset

    def get_input_and_label(self, data):
        return data

    @property
    def train_evaluations(self):
        return super(TUBerlinClassificationModel, self).train_evaluations + [AccuracyEvaluation]

    @property
    def validation_evaluations(self):
        return super(TUBerlinClassificationModel, self).validation_evaluations + [AccuracyEvaluation]
