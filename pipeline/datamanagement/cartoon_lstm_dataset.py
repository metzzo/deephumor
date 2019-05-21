import string
from collections import namedtuple

import torch

from datamanagement.base_dataset import BaseDataset

CartoonPunchlineSample = namedtuple('Sample', ['idx', 'punchline', 'funniness'])


class CartoonLSTMDataset(BaseDataset):
    def create_item(self, row, idx):
        return CartoonPunchlineSample(
            idx=idx,
            punchline=row.punchline,
            funniness=row.funniness
        )

    def create_trafo(self, trafo):
        return None