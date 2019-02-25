import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch_dct import dct_2d, idct_2d
from torchvision import transforms

from architectures.base_model import BaseModel
from architectures.tuberlin_classification_cnn import TUBerlinClassificationModel
from datamanagement.cartoon_dataset import CartoonDataset
from evaluation.accuracy_evaluation import AccuracyEvaluation
from processing.utility import Invert

# for 26.XX%
# --train_cnn --source ../export/more_downsized_export/ --epochs 2000 --batch_size 64 --model PretrainedCNNCartoonModel --loss cel


class PretrainedCNNCartoonModel(BaseModel):
    class Network(nn.Module):
        def __init__(self):
            super(PretrainedCNNCartoonModel.Network, self).__init__()

            self.model_conv = TUBerlinClassificationModel.Network()
            #self.model_conv.load_state_dict(torch.load("../good_models/20190222_134138.501959_cnn_model_sketchanet.pth"))


            removed = list(self.model_conv.seq.children())[:-1]
            model = torch.nn.Sequential(*removed)
            #for elem in list(model.children())[:-16]:
            #    for param in elem.parameters():
            #        param.requires_grad = False
            self.model_conv = torch.nn.Sequential(model,
                                                  nn.Conv2d(512, 7, kernel_size=1, stride=1),
                                                  nn.ReLU(),)

        def forward(self, x):
            x = self.model_conv(x)
            x = x.view(x.size(0), -1)
            return x

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
            transforms.RandomResizedCrop(225, scale=(0.9, 1.1), ratio=(0.75, 1.3333333333333333), interpolation=2),
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(),
            Invert(),
            transforms.ToTensor(),
        ]

    #def load_image(self, img_name):
    #    return super(PretrainedCNNCartoonModel, self).load_image(img_name=img_name).convert('RGB')

    @property
    def Dataset(self):
        return CartoonDataset

    def get_input_and_label(self, data):
        _, image, _, labels = data
        return image, labels

    @property
    def train_evaluations(self):
        return super(PretrainedCNNCartoonModel, self).train_evaluations + [AccuracyEvaluation]

    @property
    def validation_evaluations(self):
        return super(PretrainedCNNCartoonModel, self).validation_evaluations + [AccuracyEvaluation]
