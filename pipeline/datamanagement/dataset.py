import pickle
import os
import cv2
from collections import namedtuple

from torch.utils.data import Dataset
from torchvision import transforms

from datamanagement.subset import Subset
from processing.preprocess import extract_averages
from settings import DATASET_PATH

CartoonSample = namedtuple('Sample', ['idx', 'image', 'punchline', 'funniness'])

subset2file = {
    Subset.TRAINING: os.path.join(DATASET_PATH, "train_set.p"),
    Subset.TEST: os.path.join(DATASET_PATH, "test_set.p"),
    Subset.VALIDATION: os.path.join(DATASET_PATH, "validation_set.p"),
}


class CartoonDataset(Dataset):
    def __init__(self, subset: Subset):
        """
        Creates a cartoon dataset
        :param csv_file:
        :param root_dir:
        :param transform:
        """
        file_path = subset2file[subset]
        self.root_dir = os.path.dirname(file_path)
        self.cartoon_df = pickle.load(open(file_path, "rb"))
        self.transform = None

        width, height = extract_averages(self)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(32, 32,)), # TODO: keep aspect ratio
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.cartoon_df)

    def __getitem__(self, idx):
        row = self.cartoon_df.iloc[idx]

        img_name = os.path.join(self.root_dir, row['filename'])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)

        sample = CartoonSample(
            idx=idx,
            image=image,
            punchline=row['punchline'],
            funniness=row['funniness'],
        )

        return sample
