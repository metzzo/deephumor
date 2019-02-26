import numpy as np
import torch

from evaluation.base_evaluation import BaseEvaluation


class AccuracyEvaluation(BaseEvaluation):
    def __init__(self, num, batch_size):
        super(AccuracyEvaluation, self).__init__(num=num, batch_size=batch_size)
        self.sample_count = 0
        self.true_predictions = 0

    def reset(self):
        super(AccuracyEvaluation, self).reset()
        self.true_predictions = 0
        self.sample_count = 0

    def add_entry(self, predictions, actual_label, loss):
        self.true_predictions += torch.sum(predictions == actual_label)
        self.sample_count += len(predictions)

    @property
    def accuracy(self):
        return (float(self.true_predictions) / float(self.sample_count)) * 100.0

    def __str__(self):
        return 'Accuracy: {0}%'.format(self.accuracy)
