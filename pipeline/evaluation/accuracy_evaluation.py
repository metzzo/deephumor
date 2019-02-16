import numpy as np
import torch

from evaluation.base_evaluation import BaseEvaluation


class AccuracyEvaluation(BaseEvaluation):
    def __init__(self, num):
        super(AccuracyEvaluation, self).__init__(num=0)
        self.sample_count = 0
        self.true_predictions = None

    def reset(self):
        super(AccuracyEvaluation, self).reset()
        self.true_predictions = None

    def add_entry(self, predictions, actual_label, loss):
        predicted = torch.argmax(predictions, dim=1)
        pred_size = predicted.size()[0]
        self.sample_count += pred_size

        is_correct = (predicted == actual_label).int()

        if self.true_predictions is not None:
            true_pred_size = self.true_predictions.size()[0]
            if pred_size < true_pred_size:
                # increase size of is_correct
                # this might not work if there are multiple GPUs
                new_is_correct = torch.zeros(true_pred_size, dtype=torch.int32).cuda()
                new_is_correct[0:pred_size] = is_correct
                is_correct = new_is_correct

            self.true_predictions += is_correct
        else:
            self.true_predictions = is_correct

    @property
    def accuracy(self):
        return (float(self.true_predictions.sum()) / float(self.sample_count)) * 100.0

    def __str__(self):
        return 'Accuracy: {0}%'.format(self.accuracy)
