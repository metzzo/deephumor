import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from torch import tensor
from torch.nn import Softmax

from models.model import Model


class CnnClassifier(Model):
    '''
    Wrapper around a PyTorch CNN for classification.
    The network must expect inputs of shape NCHW with N being a variable batch size,
    C being the number of (image) channels, H being the (image) height, and W being the (image) width.
    The network must end with a linear layer with num_classes units (no softmax).
    The cross-entropy loss and SGD are used for training.
    '''

    def __init__(self, net: nn.Module, input_shape: tuple, num_classes: int, lr: float, wd: float):
        '''
        Ctor.
        net is the cnn to wrap. see above comments for requirements.
        input_shape is the expected input shape, i.e. (0,C,H,W).
        num_classes is the number of classes (> 0).
        lr: learning rate to use for training (sgd with Nesterov momentum of 0.9).
        wd: weight decay to use for training.
        '''

        self._net = net
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._lr = lr
        self._wd = wd
        self._softmax = Softmax(dim=0)

        self._loss = nn.L1Loss(reduction='mean')
        self._optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)

    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple.
        '''

        return self._input_shape

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''

        return (self._num_classes,)

    def train(self, data: tensor, labels: tensor) -> float:
        self._net.train()

        self._optimizer.zero_grad()
        output = self._net(data)

        loss = self._loss(output, labels.float())
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def predict(self, data: tensor) -> np.ndarray:
        self._net.eval()

        result = self._net(data)
        return result.detach()

    def save(self, path: str):
        torch.save(self._net.state_dict(), path)

    def load(self, path: str):
        self._net.load_state_dict(torch.load(path))
        self._net.eval()
