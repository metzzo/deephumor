from architectures.object_classification_cnn import ObjectClassificationModel
from architectures.pretrained_cnn import PretrainedCNNCartoonModel
from architectures.simple_regression_cnn import SimpleRegressionCNNCartoonModel
from architectures.tuberlin_classification_cnn import TUBerlinClassificationModel
from architectures.simple_classification_cnn import SimpleClassificationCNNCartoonModel


def get_model(model_name):
    models = [
        SimpleRegressionCNNCartoonModel,
        SimpleClassificationCNNCartoonModel,
        PretrainedCNNCartoonModel,
        TUBerlinClassificationModel,
        ObjectClassificationModel,
    ]
    return next((x for x in models if x.__name__ == model_name), None)()
