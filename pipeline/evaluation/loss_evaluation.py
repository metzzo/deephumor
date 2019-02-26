from evaluation.base_evaluation import BaseEvaluation
import numpy as np


class LossEvaluation(BaseEvaluation):
    def __init__(self, num, batch_size):
        super(LossEvaluation, self).__init__(num=num, batch_size=batch_size)
        self.entries = np.zeros(num, dtype=np.float32)

    def add_entry(self, predictions, actual_label, loss):
        self.entries[self.entry_count] = loss
        self.entry_count += 1

    @property
    def loss(self):
        losses = self.entries
        if self.entry_count != len(self.entries):
            losses = losses[:self.entry_count]
        return np.mean(losses)

    def __str__(self):
        return "Loss: {0}".format(self.loss)
