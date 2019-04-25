import argparse
import random

from evaluations.performance_evaluation import PerformanceEvaluation
from utility.load_class import class_for_name


def main():
    parser = argparse.ArgumentParser(
        description='Several machine learning experiments'
    )
    parser.add_argument(
        '--directory',
        type=str,
        default='.',
        help='The working directory, where to load the dataset and save result'
    )
    parser.add_argument(
        '--small',
        action='store_true',
        help='Use the small version of the dataset (useful for fast testing)'
    )

    parser.add_argument(
        '--use_cpu',
        action='store_true',
        help='Force using CPU.'
    )

    parser.add_argument(
        '--scaler',
        type=str,
        default='StandardScaler',
        help='Which sklearn scaler to use (e.g. MinMaxScaler, StandardScaler, etc)'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.1,
        help='Which learning rate to apply? Used by NN & XBoost.'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Number of elements per batch'
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of epochs of gradient descent'
    )

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.001,
        help='How much weight decay should be applied'
    )

    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='How much momentum is applied during gradient descent'
    )

    parser.add_argument(
        '--steplr_step_size',
        type=int,
        default=10,
        help='Step size to reduce learning rate'
    )

    parser.add_argument(
        '--steplr_gamma',
        type=float,
        default=0.1,
        help='Gamma of step lr'
    )

    parser.add_argument(
        '--experiment',
        type=str,
        help='What experiment to execute'
    )

    parser.add_argument(
        '--validate',
        type=bool,
        default=True,
        help='Do validation?',
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='How much of the data should be used for training.'
    )
    parser.add_argument(
        '--C',
        type=float,
        default=1.0,
        help='C parameter for SVM. The higher noise, the lower it should be.'
    )

    parser.add_argument(
        '--kernel',
        type=str,
        default='rbf',
        help='The kernel to use for the SVM'
    )
    parser.add_argument(
        '--gamma',
        type=str,
        default='scale',
        help='Kernel coefficient for SVM',
    )
    parser.add_argument(
        '--degree',
        type=int,
        default=3,
        help='degree of the polynomial kernel for SVM'
    )
    parser.add_argument(
        '--coef0',
        type=float,
        default=0.0,
        help='Independent term in kernel function for SVM'
    )
    parser.add_argument(
        '--max_depth',
        type=int,
        default=3,
        help='Maximum depth of the learned tree'
    )
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=100,
        help='Number of trees to fit.'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=1,
        help='Number of parallel jobs for XGBoost and CPU threads if running on CPU for neural networks',
    )
    parser.add_argument(
        '--min_child_weight',
        type=int,
        default=1,
        help='Minimum sum of instance weight(hessian) needed in a child'
    )
    parser.add_argument(
        '--max_delta_step',
        type=int,
        default=0,
        help='Maximum delta step we allow each tree’s weight estimation to be'
    )
    parser.add_argument(
        '--subsample',
        type=float,
        default=1,
        help='Subsample ratio of the training instance.'
    )
    parser.add_argument(
        '--colsample_bytree',
        type=float,
        default=1,
        help='Subsample ratio of columns when constructing each tree.'
    )
    parser.add_argument(
        '--colsample_bylevel',
        type=float,
        default=1,
        help='Subsample ratio of columns for each level.'
    )
    parser.add_argument(
        '--colsample_bynode',
        type=float,
        default=1,
        help='Subsample ratio of columns for each split.'
    )
    parser.add_argument(
        '--reg_alpha',
        type=float,
        default=0,
        help='L1 regularization term on weights'
    )
    parser.add_argument(
        '--reg_lambda',
        type=float,
        default=1,
        help='L2 regularization term on weights'
    )
    parser.add_argument(
        '--scale_pos_weight',
        type=float,
        default=1,
        help='Balancing of positive and negative weights.'
    )
    parser.add_argument(
        '--importance_type',
        type=str,
        default='gain',
        help='The feature importance type for the feature_importances'
    )

    parser.add_argument(
        '--baseline',
        type=bool,
        default=False,
        help='Do the experiment with baseline'
    )

    args = parser.parse_args()

    experiment = class_for_name('experiment', args.experiment)(args)
    if args.baseline:
        print("Get Baseline")

        dataset = experiment.dataset
        num_classes = dataset.num_classes

        vl = dataset.validation_label.copy()
        vl[dataset.class_column_name] = vl[dataset.class_column_name].apply(lambda x: random.randint(0, num_classes - 1))
        pe = PerformanceEvaluation(num=None, experiment=experiment)
        pe.add_entry(
            predictions=vl.values.flatten(),
            actual_label=dataset.validation_label.values.flatten(),
            loss=-1
        )
        print(pe)
    else:
        print('Train Phase: ')
        experiment.train()
        print('Test Phase: ')
        experiment.test()


if __name__ == "__main__":
    main()
