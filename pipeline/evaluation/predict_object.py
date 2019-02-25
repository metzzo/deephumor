"""
This file is used to transform the training set to a CSV file.
This allows analysis in Excel.
"""
import argparse
import pickle

from datamanagement.subset import Subset, get_subset


def setup_predict_object(parser: argparse.ArgumentParser, group):
    group.add_argument('--predict_object', action="store_true")

    def predict_object(args, device):
        if not args.predict_object:
            return
        file_path = get_subset(dataset_path=args.source, subset=Subset.VALIDATION)
        df = pickle.load(open(file_path, "rb"))
        df.to_csv(path_or_buf='./validation_set.csv')
        print("Finished!")

    return predict_object
