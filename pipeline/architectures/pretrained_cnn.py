import torch.nn as nn
import torch.nn.functional as F
import torchvision

from architectures.base_nn import BaseNN


class PretrainedCNNCartoonModel(BaseNN):
    @property
    def optimization_parameters(self):
        return self.fc.parameters()

    def __init__(self):
        super(PretrainedCNNCartoonModel, self).__init__()

        model_conv = torchvision.models.resnet18(pretrained=True)
        for param in model_conv.parameters():
            param.requires_grad = False

        num_ftrs = model_conv.fc.in_features
        self.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = self.model_conv(x)
        x = self.fc(x)

        return x
