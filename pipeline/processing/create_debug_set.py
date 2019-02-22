"""
This file is used to transform the training set to a CSV file.
This allows analysis in Excel.
"""
import argparse
import pickle
import os
import shutil

from datamanagement.cartoon_dataset import CartoonDataset
from datamanagement.subset import Subset, get_subset


def setup_create_debug_set(parser: argparse.ArgumentParser, group):
    group.add_argument('--create_debug_set', action="store_true")

    parser.add_argument("--size", type=float, default=2)

    def create_debug_set(args, device):
        if not args.create_debug_set:
            return
        dataset_path = args.source
        file_path = get_subset(dataset_path=dataset_path, subset=Subset.TRAINING)
        target_file_path = os.path.join(dataset_path, 'debug_set.p')
        cartoon_df = pickle.load(open(file_path, "rb"))

        # sample n
        cartoon_df = cartoon_df.iloc[:args.size]

        if os.path.exists(target_file_path):
            os.remove(target_file_path)

        pickle.dump(cartoon_df, open(target_file_path, "wb"))

        print("Finished!")

    return create_debug_set
