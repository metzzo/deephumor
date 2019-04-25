class BaseClassifier(object):
    def __init__(self, args):
        super(BaseClassifier, self).__init__(args)

    def train(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()
