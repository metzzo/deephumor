from evaluation.accuracy_evaluation import AccuracyEvaluation
from evaluation.base_evaluation import BaseEvaluation
from evaluation.loss_evaluation import LossEvaluation
from evaluation.mae_evaluation import MAEEvaluation


class OverallEvaluation(BaseEvaluation):
    def __init__(self, num, ignore_loss=True):
        super(OverallEvaluation, self).__init__(num=num)
        self.evaluations = [
            MAEEvaluation(num=num),
            #AccuracyEvaluation(num=num)
        ]
        if not ignore_loss:
            self.evaluations.append(LossEvaluation(num=num))

    def add_entry(self, predictions, actual_label, loss):
        for eval in self.evaluations:
            eval.add_entry(predictions=predictions, actual_label=actual_label, loss=loss)

    def __str__(self):
        output = map(str, self.evaluations)
        return '\n'.join(output)
