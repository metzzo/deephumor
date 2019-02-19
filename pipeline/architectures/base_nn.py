from torch import nn


class BaseNN(nn.Module):
    @property
    def optimization_parameters(self):
        raise NotImplementedError()

    def get_predictions(self, outputs):
        raise NotImplementedError()

