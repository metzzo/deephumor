import numpy as np
import torch

from evaluation.base_evaluation import BaseEvaluation


class MAEEvaluation(BaseEvaluation):
    def __init__(self, num, batch_size):
        super(MAEEvaluation, self).__init__(num=num * batch_size, batch_size=batch_size)

        self.reset()

    def add_entry(self, predictions, actual_label, loss):
        mae = torch.abs(actual_label - predictions).cpu()
        self.entries[self.entry_count:self.entry_count + len(mae)] = mae
        self.entry_count += len(mae)

    @property
    def MAE(self):
        return np.mean(self.entries[:self.entry_count])

    def __str__(self):
        return 'Mean Absolute Error: {0}'.format(self.MAE)
