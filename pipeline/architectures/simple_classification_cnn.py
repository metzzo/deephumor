import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from architectures.base_nn import BaseNN


class SimpleClassificationCNNCartoonModel(BaseNN):
    def get_predictions(self, outputs):
        return torch.max(outputs, 1)[1]

    @property
    def optimization_parameters(self):
        return self.parameters()

    @property
    def final_size(self):
        start_width, start_height = 141, 174
        return int(int(start_width / 2) / 2) * int(int(start_height / 2) / 2)

    def __init__(self):
        # 279x345
        super(SimpleClassificationCNNCartoonModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 200, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(200, 200, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(200 * self.final_size, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x), inplace=True))
        x = self.pool(F.relu(self.conv2(x), inplace=True))
        x = x.view(-1, 200 * self.final_size)
        x = self.fc1(x)

        return x
