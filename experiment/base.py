class BaseExperiment(object):
    def __init__(self, args):
        self.args = args

    def train(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()
