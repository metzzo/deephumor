import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from architectures.base_model import BaseModel


class PretrainedCNNCartoonModel(BaseModel):
    class Network(nn.Module):
        def __init__(self):
            super(PretrainedCNNCartoonModel.Network, self).__init__()

            self.model_conv = torchvision.models.resnet18(pretrained=True)
            for param in self.model_conv.parameters():
                param.requires_grad = False

            num_ftrs = self.model_conv.fc.in_features
            self.model_conv.fc = nn.Linear(num_ftrs, 7)

        def forward(self, x):
            x = self.model_conv(x)

            return x

    def get_predictions(self, outputs):
        return torch.max(outputs, 1)[1]

    @property
    def get_network_class(self):
        return PretrainedCNNCartoonModel.Network

    @property
    def optimization_parameters(self):
        return self.network.model_conv.fc.parameters()


