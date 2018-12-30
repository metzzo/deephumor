import numpy as np


class Evaluation(object):
    def __init__(self, num):
        self._num = num
        self.losses = np.zeros(num, dtype=np.int32)
        self.count = 0
        self.true_predictions = 0

    def add_entry(self, predicted_label, actual_label, loss):
        self.losses[self.count] = loss
        self.count += 1
        self.true_predictions += 1 if predicted_label == actual_label else 0

    @property
    def accuracy(self):
        return self.true_predictions / self.count

    @property
    def loss(self):
        losses = self.losses
        if self.count != len(self.losses):
            losses = self.losses[:self.count]
        return losses.mean()

    def __str__(self):
        return '\tAccuracy: {0}\nLoss: {1}'.format(self.accuracy, self.loss)
