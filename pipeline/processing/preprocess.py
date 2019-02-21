import argparse
import os
import cv2
import time
from progress.spinner import Spinner

from datamanagement.dataset import get_subset, CartoonDataset
from datamanagement.subset import Subset
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
from shutil import copyfile

import pickle
from functools import partial

from torch.utils.data import Dataset
import os


def extract_average_image_size(dataset: Dataset, target_folder):
    target_file = os.path.join(target_folder, 'average_image_size.p')

    try:
        avg_width, avg_height = pickle.load(open(target_file, "rb"))
    except FileNotFoundError:
        def aspect_ratio_of(sample):
            return float(sample.image.shape[0]) / sample.image.shape[1]

        def shape_of(sample, at):
            return sample.image.shape[at]

        relevant_dataset = list(filter(lambda sample: 1.2 < aspect_ratio_of(sample) < 1.3, dataset))

        widths = list(map(partial(shape_of, at=1), relevant_dataset))
        heights = list(map(partial(shape_of, at=0), relevant_dataset))

        avg_width = int(sum(widths) / len(relevant_dataset))
        avg_height = int(sum(heights) / len(relevant_dataset))

        pickle.dump([avg_width, avg_height], open(target_file, "wb"))
    print('Average Width: {0} Height: {1}'.format(avg_width, avg_height))
    return avg_width, avg_height



def setup_preprocess(parser: argparse.ArgumentParser, group):
    group.add_argument('--preprocess', action="store_true")
    parser.add_argument("--factor", type=float, default=2)

    def preprocess(args, device):
        if not args.preprocess:
            return

        target_size = extract_average_image_size(
            dataset=CartoonDataset(file_path=get_subset(dataset_path=args.source, subset=Subset.TRAINING)),
            target_folder=args.source
        )
        downsize_images(
            source_folder=args.source,
            target_folder=args.target,
            target_size=target_size,
            factor=args.factor,
        )

    return preprocess


def downsize_images(source_folder, target_folder, target_size, factor):
    print("From {0} to {1} with size {2}".format(source_folder, target_folder, target_size))

    input_directory = os.fsencode(source_folder)

    target_width, target_height = target_size

    target_width = int(target_width / factor)
    target_height = int(target_height / factor)

    if target_width % 2 == 1: target_width -= 1
    if target_height % 2 == 1: target_height -= 1

    trafo = transforms.Compose([
        transforms.Resize(size=(target_height, target_width))
    ])

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    copyfile(os.path.join(source_folder, 'train_set.p'), os.path.join(target_folder, 'train_set.p'))
    copyfile(os.path.join(source_folder, 'validation_set.p'), os.path.join(target_folder, 'validation_set.p'))

    spinner = Spinner('Loading ')
    for file in os.listdir(input_directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            original = Image.open(os.path.join(source_folder, filename))
            modified = trafo(original)

            """
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(original, cmap='gray', interpolation='bicubic')
            plt.xticks([]), plt.yticks([])
            fig.add_subplot(1, 2, 2)
            plt.imshow(modified, cmap='gray', interpolation='bicubic')
            plt.xticks([]), plt.yticks([])
            plt.show()
            time.sleep(0)
            """

            modified.save(os.path.join(target_folder, filename))
            spinner.next()

    print("Finished preprocessing")



