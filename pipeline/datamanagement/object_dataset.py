import pickle
from collections import namedtuple

from albumentations import Compose
from albumentations.augmentations import transforms

from datamanagement.base_dataset import BaseDataset
from processing.utility import imshow
import os

ObjectSample = namedtuple('Sample', ['image', 'cl'])


class ObjectDataset(BaseDataset):
    Classes = []

    def __init__(self, *args, **kwargs):
        super(ObjectDataset, self).__init__(*args, **kwargs)
        ObjectDataset.Classes = pickle.load(open(os.path.join(self.root_dir, 'classes.p'), "rb"))

    def create_item(self, row, idx):
        img = self.get_image(row['filename'])
        #imshow(img)
        return ObjectSample(
            cl=ObjectDataset.Classes.index(row['cl']),
            image=img,
        )

    def create_trafo(self, trafo):
        augmentation = Compose(trafo if len(trafo) > 0 else self.model.get_transformation())

        def transform(image):
            result = augmentation(image=image)
            return result['image'][None, :, :]  # MxN => 1xMxN

        return transform
