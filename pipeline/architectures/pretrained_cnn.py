import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch_dct import dct_2d, idct_2d
from torchvision import transforms

from architectures.base_model import BaseModel
from datamanagement.cartoon_dataset import CartoonDataset


class PretrainedCNNCartoonModel(BaseModel):
    class Network(nn.Module):
        def __init__(self):
            super(PretrainedCNNCartoonModel.Network, self).__init__()

            self.model_conv = torchvision.models.resnet18(pretrained=True)
            for param in self.model_conv.layer1.parameters():
                param.requires_grad = False
            for param in self.model_conv.layer2.parameters():
                param.requires_grad = False
            for param in self.model_conv.layer3.parameters():
                param.requires_grad = False

            num_ftrs = self.model_conv.fc.in_features
            self.model_conv.fc = nn.Linear(num_ftrs, 500)
            self.fc2 = nn.Linear(500, 7)

        def forward(self, x):
            x = F.relu(self.model_conv(x))
            x = self.fc2(x)
            return x

    def get_predictions(self, outputs):
        return torch.max(outputs, 1)[1]

    @property
    def get_network_class(self):
        return PretrainedCNNCartoonModel.Network

    @property
    def optimization_parameters(self):
        return self.network.model_conv.fc.parameters()

    def get_custom_transformation(self):
        return [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
            transforms.RandomRotation(16),
            transforms.RandomHorizontalFlip(),
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

    def load_image(self, img_name):
        return super(PretrainedCNNCartoonModel, self).load_image(img_name=img_name).convert('RGB')

    @property
    def Dataset(self):
        raise CartoonDataset

    def get_input_and_label(self, data):
        _, image, _, labels = data
        return image, labels
