from evaluations.performance_evaluation import PerformanceEvaluation
from evaluations.base_evaluation import BaseEvaluation


class OverallEvaluation(BaseEvaluation):
    @property
    def accuracy_evaluation(self):
        for eval in self.evaluations:
            if isinstance(eval, PerformanceEvaluation):
                return eval

    def __init__(self, num, experiment):
        super(OverallEvaluation, self).__init__(experiment=experiment)
        self.num = num
        self.evaluations = []

    def add_evaluations(self, evaluations):
        self.evaluations += list(map(lambda e: e(num=self.num, experiment=self.experiment), evaluations))
        self.reset()
        return self

    def add_entry(self, **kwargs):
        for eval in self.evaluations:
            eval.add_entry(**kwargs)

    def reset(self):
        super(OverallEvaluation, self).reset()
        if hasattr(self, 'evaluations'):
            for eval in self.evaluations:
                eval.reset()

    def __str__(self):
        output = map(lambda x: '\t' + str(x), self.evaluations)
        return '\n'.join(output)
