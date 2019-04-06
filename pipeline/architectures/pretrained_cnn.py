import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from architectures.base_model import BaseModel
from datamanagement.cartoon_dataset import CartoonDataset
from evaluation.accuracy_evaluation import AccuracyEvaluation


class PretrainedCNNCartoonModel(BaseModel):
    class Network(nn.Module):
        def __init__(self):
            super(PretrainedCNNCartoonModel.Network, self).__init__()

            self.model_conv = torchvision.models.resnet18()

            for param in self.model_conv.layer1.parameters():
                param.requires_grad = True
            for param in self.model_conv.layer2.parameters():
                param.requires_grad = False
            for param in self.model_conv.layer3.parameters():
                param.requires_grad = False
            for param in self.model_conv.layer4.parameters():
                param.requires_grad = False

            self.fc2 = nn.Linear(1000, 8)

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
        return PretrainedCNNCartoonModel.Network

    @property
    def optimization_parameters(self):
        return self.network.model_conv.parameters()

    def get_train_transformation(self):
        return [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
            transforms.RandomRotation(16),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]

    def get_validation_transformation(self):
        return [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]

    def load_image(self, img_name):
        return super(PretrainedCNNCartoonModel, self).load_image(img_name=img_name).convert('RGB')

    @property
    def Dataset(self):
        return CartoonDataset

    def get_input_and_label(self, data):
        _, image, labels = data
        return image, labels
    @property
    def train_evaluations(self):
        return super(PretrainedCNNCartoonModel, self).train_evaluations + [AccuracyEvaluation]

    @property
    def validation_evaluations(self):
        return super(PretrainedCNNCartoonModel, self).validation_evaluations + [AccuracyEvaluation]


