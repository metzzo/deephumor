import numpy as np


import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import transforms

from cnn_experiments.base_model import BaseCNNModel
from datamanagement.cartoon_cnn_dataset import CartoonCNNDataset
from evaluation.mae_evaluation import MAEEvaluation


class SimpleRegressionCNNCartoonCNNModel(BaseCNNModel):
    class Network(nn.Module):
        @property
        def final_size(self):
            start_width, start_height = 141, 174
            return int(int(int(int(start_width / 2) / 2) / 2) / 2) * int(int(int(int(start_height / 2) / 2) / 2) / 2)

        def __init__(self):
            # 279x345
            super(SimpleRegressionCNNCartoonCNNModel.Network, self).__init__()

            self.conv1 = nn.Conv2d(1, 200, kernel_size=5, padding=1)
            self.conv2 = nn.Conv2d(200, 200, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(200, 200, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(200, 200, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(200 * self.final_size, 1)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x), inplace=True))
            x = self.pool(F.relu(self.conv2(x), inplace=True))
            x = self.pool(F.relu(self.conv3(x), inplace=True))
            x = self.pool(F.relu(self.conv4(x), inplace=True))
            x = x.view(-1, 200 * self.final_size)
            x = self.fc1(x)

            return x

    @property
    def get_network_class(self):
        return SimpleRegressionCNNCartoonCNNModel.Network

    def get_predictions(self, outputs):
        return outputs.detach().flatten()

    def get_labels(self, labels):
        return labels.float()

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
        return super(SimpleRegressionCNNCartoonCNNModel, self).train_evaluations + [MAEEvaluation]

    @property
    def validation_evaluations(self):
        return super(SimpleRegressionCNNCartoonCNNModel, self).validation_evaluations + [MAEEvaluation]

