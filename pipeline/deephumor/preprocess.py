import os
import pickle
from functools import partial

from progress.bar import Bar

from pipeline.deephumor.dataset import CartoonDataset
from pipeline.settings import DATASET_PATH


def get_average_image_size():
    try:
        avg_width, avg_height = pickle.load(open('average_image_size.p', "rb"))
    except FileNotFoundError:
        dataset = CartoonDataset(
            file_path=DATASET_PATH
        )

        def aspect_ratio_of(sample):
            return float(sample.image.shape[0]) / sample.image.shape[1]

        def shape_of(sample, at):
            return sample.image.shape[at]

        relevant_dataset = list(filter(lambda sample: 1.2 < aspect_ratio_of(sample) < 1.3, dataset))

        widths = list(map(partial(shape_of, at=1), relevant_dataset))
        heights = list(map(partial(shape_of, at=0), relevant_dataset))

        avg_width = int(sum(widths) / len(relevant_dataset))
        avg_height = int(sum(heights) / len(relevant_dataset))

        pickle.dump([avg_width, avg_height], open('average_image_size.p', "wb"))
    print('Average Width: {0} Height: {1}'.format(avg_width, avg_height))
    return avg_width, avg_height


if __name__ == '__main__':
    get_average_image_size()