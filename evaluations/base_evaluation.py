class BaseEvaluation(object):
    def __init__(self, experiment):
        self.experiment = experiment
        self.reset()

    def reset(self):
        pass

    def add_entry(self, predictions, actual_label, loss):
        raise NotImplementedError()
