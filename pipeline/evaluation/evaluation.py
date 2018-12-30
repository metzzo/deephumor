import numpy as np
import torch


class Evaluation(object):
    def __init__(self, num, batch_size, ignore_loss:bool=False):
        self.losses = np.zeros(num, dtype=np.int32)
        self.loss_count = 0
        self.sample_count = 0
        self.true_predictions = None
        self.ignore_loss = ignore_loss

        self.reset()

    def reset(self):
        self.loss_count = 0
        self.sample_count = 0
        self.true_predictions = None

    def add_entry(self, predictions, actual_label, loss=0):
        if not self.ignore_loss:
            self.losses[self.loss_count] = loss
        self.loss_count += 1

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

    @property
    def loss(self):
        losses = self.losses
        if self.loss_count != len(self.losses):
            losses = self.losses[:self.loss_count]
        return losses.mean()

    def __str__(self):
        if self.ignore_loss:
            return '\tAccuracy: {0}'.format(self.accuracy)
        else:
            return '\tAccuracy: {0}\n\tLoss: {1}'.format(self.accuracy, self.loss)
