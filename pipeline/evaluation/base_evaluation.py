import numpy as np


class BaseEvaluation(object):
    def __init__(self, num, batch_size):
        self.entries = np.zeros(num, dtype=np.float32)
        self.entry_count = 0
        self.reset()

    def reset(self):
        self.entry_count = 0

    def add_entry(self, predictions, actual_label, loss):
        raise NotImplementedError()
