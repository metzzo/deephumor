from functools import partial

import numpy as np
import torch

from evaluation.base_evaluation import BaseEvaluation


class DetailedEvaluation(BaseEvaluation):
    def __init__(self, model, num, batch_size):
        super(DetailedEvaluation, self).__init__(num=num, batch_size=batch_size)
        self.predictions = []
        self.actual_labels = []
        self.model = model

    def reset(self):
        super(DetailedEvaluation, self).reset()
        self.predictions = []
        self.actual_labels = []

    def add_entry(self, predictions, actual_label, loss):
        self.predictions += list(predictions.cpu().numpy())
        self.actual_labels += list(actual_label.cpu().numpy())

    def __str__(self):
        results = map(
            lambda x: '{0}\t{1}'.format(*x),
            zip(
                map(partial(self.model.get_human_readable_class, is_predicted=True), self.predictions),
                map(partial(self.model.get_human_readable_class, is_predicted=False), self.actual_labels)
            )
        )
        return 'Detailed Results:\n\tPredicted\tActual Label\n\t{0}'.format('\n\t'.join(results))
