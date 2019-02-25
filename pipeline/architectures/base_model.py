from PIL import Image
from torchvision import transforms


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

    def get_train_transformation(self):
        return [
            transforms.ToTensor(),
        ]

    def get_validation_transformation(self):
        raise [
            transforms.ToTensor(),
        ]

    def load_image(self, img_name):
        return Image.open(img_name)

    @property
    def Dataset(self):
        raise NotImplementedError()

    def get_input_and_label(self, data):
        raise NotImplementedError()
