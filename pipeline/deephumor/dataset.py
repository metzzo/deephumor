import pickle
import os
import cv2
from collections import namedtuple

from torch.utils.data import Dataset

CartoonSample = namedtuple('Sample', ['idx', 'image', 'punchline', 'funniness'])


class CartoonDataset(Dataset):
    def __init__(self, file_path):
        """
        Creates a cartoon dataset
        :param csv_file:
        :param root_dir:
        :param transform:
        """
        self.cartoon_df = pickle.load(open(file_path, "rb"))
        self.root_dir = os.path.dirname(file_path)

    def __len__(self):
        return len(self.cartoon_df)

    def __getitem__(self, idx):
        row = self.cartoon_df.iloc[idx]

        img_name = os.path.join(self.root_dir, row['filename'])
        image = cv2.imread(img_name)

        sample = CartoonSample(
            idx=idx,
            image=image,
            punchline=row['punchline'],
            funniness=row['funniness'],
        )

        return sample

    def get_batch_data(self, sample):
        return sample.image, sample.funniness
