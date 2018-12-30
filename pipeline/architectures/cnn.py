import numpy as np


import torch.nn as nn
import torch.nn.functional as F

from settings import USE_DROP_OUT


class SimpleCNNCartoonModel(nn.Module):
    def __init__(self):
        super(SimpleCNNCartoonModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 36, 3)
        self.conv2 = nn.Conv2d(36, 108, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(108 * 6 * 6, 200)
        self.dropout = nn.Dropout(p=0.75)
        self.fc2 = nn.Linear(200, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 108 * 6 * 6)
        if USE_DROP_OUT:
            x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
