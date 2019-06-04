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

    @property
    def use_pytorch_trafo(self):
        return True

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
