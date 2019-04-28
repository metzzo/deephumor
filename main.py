import numpy as np
from allennlp.commands.train import train_model_from_file
from allennlp.models import load_archive

from jigsaw import *
from jigsaw.eval import JigsawEvaluator
from predictor import predict, predict_test


def train():
    train_model_from_file(
        parameter_filename='./jigsaw/config.json',
        serialization_dir='./result_2/',
        recover=True,
        #force=True,
    )
    archive_path = './result_2/model.tar.gz'
    predicted, df = predict(
        archive=archive_path,
        file='../dataset/jigsaw/validation.csv'
    )
    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
    ]
    jigsaw_eval = JigsawEvaluator(
        y_true=df['target'].values,
        y_identity=df[identity_columns].values,
    )
    print("AUC-ROC Score: {}".format(jigsaw_eval.compute_bias_metrics_for_model(
        y_pred=np.array(predicted)
    )))

    predict_test()



if __name__ == '__main__':
    train()
