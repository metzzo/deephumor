import os

from xgboost import XGBClassifier
from classifier.base import BaseClassifier
from evaluations.performance_evaluation import PerformanceEvaluation
from evaluations.overall_evaluation import OverallEvaluation
from utility.test_collector import TestCollector


class XGBoostClassifierMixin(BaseClassifier):
    @property
    def decision_function_shape(self):
        return 'ovr'

    def __init__(self, args):
        super(XGBoostClassifierMixin, self).__init__(args)

        if not hasattr(self, 'dataset'):
            self.dataset = None
            raise RuntimeError("Missing dataset")

        self.clf = None

    def train(self):
        print('Train XGBoost')
        self.clf = XGBClassifier(
            learning_rate=self.args.learning_rate,
            max_depth=self.args.max_depth,
            n_estimators=self.args.n_estimators,
            n_jobs=self.args.n_jobs,
            min_child_weight=self.args.min_child_weight,
            max_delta_step=self.args.max_delta_step,
            subsample=self.args.subsample,
            colsample_bytree=self.args.colsample_bytree,
            colsample_bylevel=self.args.colsample_bylevel,
            colsample_bynode=self.args.colsample_bynode,
            reg_alpha=self.args.reg_alpha,
            reg_lambda=self.args.reg_lambda,
            scale_pos_weight=self.args.scale_pos_weight,
            importance_type=self.args.importance_type,
            verbosity=2,
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
        print('Test XGBoost')
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
