import os
import random

import torchvision.datasets as dset

import argparse
import pickle

from PIL import Image

from datamanagement.dataset import get_subset, CartoonDataset
from datamanagement.subset import Subset


def setup_prepare_mnist(parser: argparse.ArgumentParser, group):
    group.add_argument('--prepare_mnist', action="store_true")

    def prepare_mnist(args, device):
        if not args.prepare_mnist:
            return
        train_set = dset.MNIST(root='./', train=True, download=True)

        targets = {
            i: [] for i in range(1, 8)
        }

        for i in range(0, len(train_set)):
            img, target = train_set[i]
            target = int(target)
            if target in targets:
                targets[target] += [img]
        for subset in [Subset.TRAINING, Subset.VALIDATION]:
            file_path = get_subset(dataset_path=args.source, subset=subset)
            df = pickle.load(open(file_path, "rb"))
            for _, row in df.iterrows():
                funniness = row.funniness
                candidate = random.choice(targets[funniness])
                to_save = candidate.resize((141, 174), Image.ANTIALIAS)
                to_save.save(os.path.join(args.source, row.filename))
            print("Finished! " + str(subset))

    return prepare_mnist
