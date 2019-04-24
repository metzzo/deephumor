import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from cnn_experiments.base_model import BaseCNNModel
from datamanagement.cartoon_cnn_dataset import CartoonCNNDataset
from evaluation.accuracy_evaluation import AccuracyEvaluation


class SimpleClassificationCNNCartoonCNNModel(BaseCNNModel):
    class Network(nn.Module):
        @property
        def final_size(self):
            start_width, start_height = 141, 174
            return int(int(start_width / 2) / 2) * int(int(start_height / 2) / 2)

        def __init__(self):
            # 279x345
            super(SimpleClassificationCNNCartoonCNNModel.Network, self).__init__()

            self.conv1 = nn.Conv2d(1, 100, kernel_size=3, padding=1)
            self.norm = nn.BatchNorm2d(100)
            self.conv2 = nn.Conv2d(100, 100, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(100 * self.final_size, 7)
            self.dropout = nn.Dropout(p=0.5)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x), inplace=True))
            x = self.pool(F.relu(self.norm(self.conv2(x)), inplace=True))
            x = self.dropout(x)
            x = x.view(-1, 100 * self.final_size)
            x = self.fc1(x)

            return x

    @property
    def get_network_class(self):
        return SimpleClassificationCNNCartoonCNNModel.Network

    def get_predictions(self, outputs):
        return torch.max(outputs, 1)[1]

    @property
    def optimization_parameters(self):
        return self.network.parameters()

    @property
    def Dataset(self):
        return CartoonCNNDataset

    def get_input_and_label(self, data):
        _, image, labels = data
        return image, labels

    @property
    def train_evaluations(self):
        return super(SimpleClassificationCNNCartoonCNNModel, self).train_evaluations + [AccuracyEvaluation]

    @property
    def validation_evaluations(self):
        return super(SimpleClassificationCNNCartoonCNNModel, self).validation_evaluations + [AccuracyEvaluation]
