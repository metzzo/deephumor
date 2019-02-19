from torch import nn


class BaseModel(object):
    def __init__(self):
        self.network = self.get_network_class()

    @property
    def optimization_parameters(self):
        raise NotImplementedError()

    @property
    def get_network_class(self):
        raise NotImplementedError()

    def get_predictions(self, outputs):
        raise NotImplementedError()

    def get_labels(self, labels):
        return labels
