class BaseEvaluation(object):
    def __init__(self, num, batch_size):
        self.entry_count = 0
        self.batch_size = batch_size
        self.num = num
        self.reset()

    def reset(self):
        self.entry_count = 0

    def add_entry(self, predictions, actual_label, loss, top_five=None):
        raise NotImplementedError()
