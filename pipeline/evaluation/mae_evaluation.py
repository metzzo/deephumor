import numpy as np
import torch

from evaluation.base_evaluation import BaseEvaluation


class MAEEvaluation(BaseEvaluation):
    def __init__(self, num):
        super(MAEEvaluation, self).__init__(num=num)

        self.reset()

    def add_entry(self, predictions, actual_label, loss):
        loss = torch.abs(actual_label - predictions)
        mean_loss = torch.mean(loss).float()
        self.entries[self.entry_count] = mean_loss
        self.entry_count += 1

    @property
    def MAE(self):
        return np.mean(self.entries[:self.entry_count])

    def __str__(self):
        return 'Mean Absolute Error: {0}'.format(self.MAE)
