from collections import namedtuple

from datamanagement.base_dataset import BaseDataset

CartoonSample = namedtuple('Sample', ['idx', 'image', 'funniness'])


class CartoonCNNDataset(BaseDataset):
    def create_item(self, row, idx):
        return CartoonSample(
            idx=idx,
            image=self.get_image(row['filename']),
            funniness=row['funniness'],
        )
