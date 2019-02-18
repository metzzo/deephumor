import pickle
import os
import cv2
from collections import namedtuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datamanagement.subset import Subset

CartoonSample = namedtuple('Sample', ['idx', 'image', 'punchline', 'funniness'])


def get_subset(dataset_path, subset: Subset):
    dict = {
        Subset.TRAINING: os.path.join(dataset_path, "train_set.p"),
        Subset.TEST: os.path.join(dataset_path, "test_set.p"),
        Subset.VALIDATION: os.path.join(dataset_path, "validation_set.p"),
    }
    return dict[subset]


class CartoonDataset(Dataset):
    def __init__(self, file_path):
        """
        Creates a cartoon dataset
        :param csv_file:
        :param root_dir:
        :param transform:
        """
        self.root_dir = os.path.dirname(file_path)
        self.cartoon_df = pickle.load(open(file_path, "rb"))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.224])
        ])

    def __len__(self):
        return len(self.cartoon_df)

    def __getitem__(self, idx):
        row = self.cartoon_df.iloc[idx]

        img_name = os.path.join(self.root_dir, row['filename'])
        image = Image.open(img_name)
        if self.transform is not None:
            image = self.transform(image)

        sample = CartoonSample(
            idx=idx,
            image=image,
            punchline=row['punchline'],
            funniness=(row['funniness'] - 1) / 7.0,
        )

        return sample
