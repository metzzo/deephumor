from torchvision import transforms

from architectures.tuberlin_classification_cnn import TUBerlinClassificationModel
from datamanagement.object_dataset import ObjectDataset


class ObjectClassificationModel(TUBerlinClassificationModel):
    @property
    def Dataset(self):
        return ObjectDataset

    def get_validation_transformation(self):
        return [
            transforms.Resize((225, 225)),
            transforms.ToTensor(),
        ]
