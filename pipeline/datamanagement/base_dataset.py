import os
import pickle

from albumentations import Compose
from torch.utils.data import Dataset
from torchvision import transforms

from cnn_experiments.base_model import BaseCNNModel


class BaseDataset(Dataset):
    class EmptyCNNModel(BaseCNNModel):
        @property
        def get_network_class(self):
            return lambda: None

    @property
    def use_pytorch_trafo(self):
        return True

    def __init__(self, file_path, model=None, trafo=None):
        self.root_dir = os.path.dirname(file_path)
        self.df = pickle.load(open(file_path, "rb"))
        self.model = model if model else BaseDataset.EmptyCNNModel()

        trafo = [] if trafo is None else trafo
        self.transform = self.create_trafo(trafo=trafo)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = self.create_item(row=row, idx=idx)
        return sample

    def create_item(self, row, idx):
        raise NotImplementedError()

    def get_image(self, filename):
        img_name = os.path.join(self.root_dir, filename)
        image = self.model.load_image(img_name)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def create_trafo(self, trafo):
        if self.use_pytorch_trafo:
            return transforms.Compose(trafo if len(trafo) > 0 else self.model.get_transformation())
        else:
            augmentation = Compose(trafo if len(trafo) > 0 else self.model.get_transformation())

            def transform(image):
                result = augmentation(image=image)
                return result['image'][None, :, :]  # MxN => 1xMxN

            return transform
