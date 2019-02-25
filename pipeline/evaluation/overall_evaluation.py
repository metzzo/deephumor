from evaluation.accuracy_evaluation import AccuracyEvaluation
from evaluation.base_evaluation import BaseEvaluation


class OverallEvaluation(BaseEvaluation):
    @property
    def accuracy_evaluation(self):
        for eval in self.evaluations:
            if isinstance(eval, AccuracyEvaluation):
                return eval

    def __init__(self, num, batch_size):
        super(OverallEvaluation, self).__init__(num=num, batch_size=batch_size)
        self.evaluations = []

    def add_evaluations(self, evaluations):
        self.evaluations += list(map(lambda e: e(num=self.num, batch_size=self.batch_size), evaluations))
        self.reset()
        return self

    def add_entry(self, predictions, actual_label, loss):
        for eval in self.evaluations:
            eval.add_entry(predictions=predictions, actual_label=actual_label, loss=loss)

    def reset(self):
        super(OverallEvaluation, self).reset()
        if hasattr(self, 'evaluations'):
            for eval in self.evaluations:
                eval.reset()

    def __str__(self):
        output = map(lambda x: '\t' + str(x), self.evaluations)
        return '\n'.join(output)
