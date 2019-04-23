from cnn_experiments.object_classification_cnn import ObjectClassificationModel
from cnn_experiments.pretrained_cnn import PretrainedCNNCartoonCNNModel
from cnn_experiments.resnet_classification_cnn import ResNetClassificationCNNModel
from cnn_experiments.simple_regression_cnn import SimpleRegressionCNNCartoonCNNModel
from cnn_experiments.tuberlin_classification_cnn import TUBerlinClassificationCNNModel
from cnn_experiments.simple_classification_cnn import SimpleClassificationCNNCartoonCNNModel


def get_model(model_name):
    models = [
        SimpleRegressionCNNCartoonCNNModel,
        SimpleClassificationCNNCartoonCNNModel,
        PretrainedCNNCartoonCNNModel,
        TUBerlinClassificationCNNModel,
        ObjectClassificationModel,
        ResNetClassificationCNNModel,
    ]
    return next((x for x in models if x.__name__ == model_name), None)()
