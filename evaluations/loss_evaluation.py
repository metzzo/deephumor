from evaluations.base_evaluation import BaseEvaluation
import numpy as np


class LossEvaluation(BaseEvaluation):
    def __init__(self, num, experiment):
        super(LossEvaluation, self).__init__(experiment=experiment)
        self.entries = np.zeros(num, dtype=np.float32)
        self.entry_count = 0

    def add_entry(self, predictions, actual_label, loss, top_five=None):
        self.entries[self.entry_count] = loss
        self.entry_count += 1

    def reset(self):
        self.entry_count = 0

    @property
    def loss(self):
        losses = self.entries
        if self.entry_count != len(self.entries):
            losses = losses[:self.entry_count]
        return np.mean(losses)

    def __str__(self):
        return "Loss: {0}".format(self.loss)
