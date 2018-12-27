import pickle
import os

from torch.utils.data import Dataset

from skimage import io, transform

class CartoonDataset(Dataset):
    def __init__(self, file_path, transform=None):
        """
        Creates a cartoon dataset
        :param csv_file:
        :param root_dir:
        :param transform:
        """
        self.cartoon_df = pickle.load(open(file_path, "rb"))
        self.root_dir = os.path.dirname(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.cartoon_df)

    def __getitem__(self, idx):
        row = self.cartoon_df.iloc[idx]

        img_name = os.path.join(self.root_dir, row['filename'])
        image = io.imread(img_name)

        sample = {
            'image': image,
            'punchline': row['punchline'],
            'funniness': row['funniness'],
        }

        if self.transform:
            sample = self.transform(sample)

        return sample