import numpy as np
import torch

from evaluation.base_evaluation import BaseEvaluation
from settings import BATCH_SIZE


class MAEEvaluation(BaseEvaluation):
    def __init__(self, num):
        super(MAEEvaluation, self).__init__(num=num * BATCH_SIZE)

        self.reset()
        self._loss = torch.nn.L1Loss(reduction='mean')

    def add_entry(self, predictions, actual_label, loss):
        # mae = torch.abs(actual_label.float() - 2/7).cpu()
        mae = torch.abs(actual_label.float() - predictions.flatten()).cpu()
        self.entries[self.entry_count:self.entry_count + len(mae)] = mae
        self.entry_count += len(mae)

    @property
    def MAE(self):
        return np.mean(self.entries[:self.entry_count]) * 7

    def __str__(self):
        return 'Mean Absolute Error: {0}'.format(self.MAE)
