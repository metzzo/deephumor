from enum import Enum


class Subset(Enum):
    '''
    Dataset subsets.
    '''

    TRAINING = 1
    VALIDATION = 2
    TEST = 3
    DEBUG = 4
