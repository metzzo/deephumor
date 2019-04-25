from functools import partial

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from evaluations.base_evaluation import BaseEvaluation


class PerformanceEvaluation(BaseEvaluation):
    def __init__(self, num, experiment):
        super(PerformanceEvaluation, self).__init__(experiment=experiment)
        self.predictions = None
        self.actual_labels = None

    def reset(self):
        super(PerformanceEvaluation, self).reset()
        self.predictions = None
        self.actual_labels = None

    def add_entry(self, predictions, actual_label, loss, top_five=None):
        self.predictions = predictions.copy() if self.predictions is None else np.concatenate(
            (self.predictions, predictions),
            axis=0
        )
        self.actual_labels = actual_label.copy() if self.actual_labels is None else np.concatenate(
            (self.actual_labels, actual_label),
            axis=0
        )

    @property
    def accuracy(self):
        return self.calc_score(score=accuracy_score)

    def calc_score(self, score):
        func = partial(score, y_true=self.actual_labels, y_pred=self.predictions)

        if self.experiment.dataset.num_classes != 2 and score != accuracy_score:
            return func(average='weighted') * 100
        else:
            return func() * 100

    @property
    def precision(self):
        return self.calc_score(score=precision_score)

    @property
    def recall(self):
        return self.calc_score(score=recall_score)

    @property
    def f1(self):
        return self.calc_score(score=f1_score)

    def __str__(self):
        return 'Accuracy: {0}%\tPrecision: {1}\tRecall: {2}\tF1: {3}'.format(
            round(self.accuracy, 2),
            round(self.precision, 2),
            round(self.recall, 2),
            round(self.f1, 2),
        )
