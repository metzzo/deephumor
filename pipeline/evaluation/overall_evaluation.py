from evaluation.accuracy_evaluation import AccuracyEvaluation
from evaluation.base_evaluation import BaseEvaluation
from evaluation.loss_evaluation import LossEvaluation
from evaluation.mae_evaluation import MAEEvaluation


class OverallEvaluation(BaseEvaluation):

    def __init__(self, num, batch_size, ignore_loss=True):
        self.accuracy_evaluation = AccuracyEvaluation(num=num, batch_size=batch_size)
        self.evaluations = [
            MAEEvaluation(num=num, batch_size=batch_size),
            self.accuracy_evaluation,
        ]
        if not ignore_loss:
            self.evaluations.append(LossEvaluation(num=num, batch_size=batch_size))

        super(OverallEvaluation, self).__init__(num=num, batch_size=batch_size)

    def add_entry(self, predictions, actual_label, loss):
        for eval in self.evaluations:
            eval.add_entry(predictions=predictions, actual_label=actual_label, loss=loss)

    def reset(self):
        super(OverallEvaluation, self).reset()
        for eval in self.evaluations:
            eval.reset()

    def __str__(self):
        output = map(lambda x: '\t' + str(x), self.evaluations)
        return '\n'.join(output)
