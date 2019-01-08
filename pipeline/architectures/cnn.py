from unittest.mock import inplace

import numpy as np


import torch.nn as nn
import torch.nn.functional as F

from settings import USE_DROP_OUT


class SimpleCNNCartoonModel(nn.Module):
    @property
    def final_size(self):
        start_width, start_height = 278, 344
        return int((int((start_width - 2) / 2) - 2) / 2) * int((int((start_height - 2) / 2) - 2) / 2)

    def __init__(self):
        # 279x345
        super(SimpleCNNCartoonModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 36, 3)
        self.conv2 = nn.Conv2d(36, 108, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(108 * self.final_size, 200)
        self.dropout = nn.Dropout(p=0.75)
        self.fc2 = nn.Linear(200, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x), inplace=True))
        x = self.pool(F.relu(self.conv2(x), inplace=True))
        x = x.view(-1, 108 * self.final_size)
        if USE_DROP_OUT:
            x = self.dropout(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)

        return x
