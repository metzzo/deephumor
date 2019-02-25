import os

from datamanagement.subset import Subset


def get_subset(dataset_path, subset: Subset):
    dict = {
        Subset.TRAINING: os.path.join(dataset_path, "train_set.p"),
        Subset.TEST: os.path.join(dataset_path, "test_set.p"),
        Subset.VALIDATION: os.path.join(dataset_path, "validation_set.p"),
        Subset.DEBUG: os.path.join(dataset_path, "debug_set.p"),
    }
    return dict[subset]
