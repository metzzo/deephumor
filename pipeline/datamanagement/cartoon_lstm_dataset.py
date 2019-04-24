import string
from collections import namedtuple

import torch

from datamanagement.base_dataset import BaseDataset

CartoonPunchlineSample = namedtuple('Sample', ['idx', 'punchline', 'funniness'])

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters).long()
    tensor[0][letter_to_index(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_tensor(letter)] = 1
    return tensor


class CartoonLSTMDataset(BaseDataset):
    def create_item(self, row, idx):
        return CartoonPunchlineSample(
            idx=idx,
            punchline=line_to_tensor(row.punchline),
            funniness=row.funniness
        )

    def create_trafo(self, trafo):
        return None