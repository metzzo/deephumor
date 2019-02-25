from collections import namedtuple

from datamanagement.base_dataset import BaseDataset

ObjectSample = namedtuple('Sample', ['image', 'cl'])


class ObjectDataset(BaseDataset):
    def create_item(self, row, idx):
        return ObjectSample(
            image=self.get_image(row['filename']),
            cl=row['cl'],
        )
