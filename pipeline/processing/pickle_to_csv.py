"""
This file is used to transform the training set to a CSV file.
This allows analysis in Excel.
"""
import argparse
import pickle

from datamanagement.dataset import get_subset, CartoonDataset
from datamanagement.subset import Subset


def setup_pickle_to_csv(parser: argparse.ArgumentParser, group):
    group.add_argument('--pickle_to_csv', action="store_true")

    def pickle_to_csv(args):
        if not args.pickle_to_csv:
            return
        file_path = get_subset(dataset_path=args.source, subset=Subset.TRAINING)
        df = pickle.load(open(file_path, "rb"))
        df.to_csv(path_or_buf='./training_set.csv')
        print("Finished!")

    return pickle_to_csv
