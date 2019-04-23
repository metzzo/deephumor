from PIL import Image
from torchvision import transforms

from evaluation.loss_evaluation import LossEvaluation


class BaseCNNModel(object):
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

    def get_train_transformation(self):
        return [
            transforms.ToTensor(),
        ]

    def get_validation_transformation(self):
        return [
            transforms.ToTensor(),
        ]

    def load_image(self, img_name):
        return Image.open(img_name)

    @property
    def Dataset(self):
        raise NotImplementedError()

    def get_input_and_label(self, data):
        raise NotImplementedError()

    @property
    def train_evaluations(self):
        return [LossEvaluation]

    @property
    def validation_evaluations(self):
        return []

    def get_human_readable_class(self, cl, is_predicted):
        return cl
