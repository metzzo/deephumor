import torch
from numpy import long
from torch.utils.data import Dataset


class PyTorchDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        data = torch.tensor(self.data.iloc[index]).float()
        if self.labels is not None:
            return (
                data,
                torch.tensor(self.labels.iloc[index].values[0]).long(),
            )
        else:
            return (
                data,
            )

    def __len__(self):
        return len(self.data)
