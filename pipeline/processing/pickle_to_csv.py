"""
This file is used to transform the training set to a CSV file.
This allows analysis in Excel.
"""
import argparse
import pickle

from datamanagement.cartoon_dataset import CartoonDataset
from datamanagement.subset import Subset, get_subset


def setup_pickle_to_csv(parser: argparse.ArgumentParser, group):
    group.add_argument('--pickle_to_csv', action="store_true")

    def pickle_to_csv(args, device):
        if not args.pickle_to_csv:
            return
        file_path = get_subset(dataset_path=args.source, subset=Subset.VALIDATION)
        df = pickle.load(open(file_path, "rb"))
        df.to_csv(path_or_buf='./validation_set.csv')
        print("Finished!")

    return pickle_to_csv
