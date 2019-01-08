import argparse
import os
import cv2
import time
from progress.spinner import Spinner

from datamanagement.dataset import get_subset, CartoonDataset
from datamanagement.subset import Subset
from processing.extract_average_image_size import extract_average_image_size
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
from shutil import copyfile


def setup_preprocess(parser: argparse.ArgumentParser, group):
    group.add_argument('--preprocess', action="store_true")

    parser.add_argument("--target", type=str)
    parser.add_argument("--factor", type=float, default=2)

    def preprocess(args):
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



