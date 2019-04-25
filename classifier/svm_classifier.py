import os

from sklearn import svm

from classifier.base import BaseClassifier
from evaluations.performance_evaluation import PerformanceEvaluation
from evaluations.overall_evaluation import OverallEvaluation
from utility.test_collector import TestCollector


class SVMClassifierMixin(BaseClassifier):
    @property
    def decision_function_shape(self):
        return 'ovr'

    def __init__(self, args):
        super(SVMClassifierMixin, self).__init__(args)

        if not hasattr(self, 'dataset'):
            self.dataset = None
            raise RuntimeError("Missing dataset")

        self.clf = None

    def train(self):
        print('Train SVM')
        self.clf = svm.SVC(
            C=self.args.C,
            gamma=self.args.gamma,
            kernel=self.args.kernel,
            degree=self.args.degree,
            coef0=self.args.coef0,
            decision_function_shape=self.decision_function_shape,
            verbose=True
        )

        train_data, train_labels = self.dataset.get_subset(
            usage='train'
        )
        train_labels = train_labels.values.flatten()

        self.clf.fit(
            train_data,
            train_labels
        )

        val_data, val_labels = self.dataset.get_subset(
            usage='val'
        )
        val_labels = val_labels.values.flatten()

        predicted = self.clf.predict(
            val_data
        )

        evaluations = OverallEvaluation(
            num=len(train_data), experiment=self
        ).add_evaluations([
            PerformanceEvaluation
        ])
        evaluations.add_entry(
            predictions=predicted,
            actual_label=val_labels,
            loss=None,
        )
        print('\n')
        print('Validation performance {0}'.format(str(evaluations)))

        predicted = self.clf.predict(
            train_data
        )

        evaluations.reset()
        evaluations.add_entry(
            predictions=predicted,
            actual_label=train_labels,
            loss=None,
        )

        print('Training performance {0}'.format(str(evaluations)))

    def test(self):
        print('Test SVM')
        test_data, test_ids = self.dataset.get_subset(
            usage='test'
        )

        predicted = self.clf.predict(
            test_data
        )

        collector = TestCollector(dataset=self.dataset)

        collector.collect_test(
            id=test_ids.values,
            predicted_class=predicted,
        )

        collector.save_csv(
            path=os.path.join(self.args.directory, 'test_result.csv')
        )
