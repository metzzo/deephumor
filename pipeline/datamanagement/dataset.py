import pickle
import os
import cv2
from collections import namedtuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from architectures.base_model import BaseModel
from datamanagement.subset import Subset

CartoonSample = namedtuple('Sample', ['idx', 'image', 'punchline', 'funniness'])


def get_subset(dataset_path, subset: Subset):
    dict = {
        Subset.TRAINING: os.path.join(dataset_path, "train_set.p"),
        Subset.TEST: os.path.join(dataset_path, "test_set.p"),
        Subset.VALIDATION: os.path.join(dataset_path, "validation_set.p"),
        Subset.DEBUG: os.path.join(dataset_path, "debug_set.p"),
    }
    return dict[subset]


class CartoonDataset(Dataset):
    class EmptyModel(BaseModel):
        pass

    def __init__(self, file_path, model=None):
        """
        Creates a cartoon dataset
        :param csv_file:
        :param root_dir:
        :param transform:
        """
        self.root_dir = os.path.dirname(file_path)
        self.cartoon_df = pickle.load(open(file_path, "rb"))
        self.model = model if model else CartoonDataset.EmptyModel()

        custom_transformation = self.model.get_custom_transformation()

        self.transform = transforms.Compose(custom_transformation if len(custom_transformation) > 0 else [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.cartoon_df)

    def __getitem__(self, idx):
        row = self.cartoon_df.iloc[idx]

        img_name = os.path.join(self.root_dir, row['filename'])
        image = self.model.load_image(img_name)
        if self.transform is not None:
            image = self.transform(image)

        sample = CartoonSample(
            idx=idx,
            image=image,
            punchline=row['punchline'],
            funniness=row['funniness'], #row['funniness'] - 1) / 7.0,
        )

        return sample
