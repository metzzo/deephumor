import numpy as np
from allennlp.models import load_archive

from jigsaw import *

ARCHIVE_PATH = './result_2/model.tar.gz'
FILE_PATH = '../dataset/jigsaw/test.csv'


def predict(archive, file):
    predictor = Predictor.from_archive(
        load_archive(archive, cuda_device=0),
        predictor_name='sentence_classifier_predictor'
    )
    print("Predictor loaded")
    df = pd.read_csv(file)
    print("CSV loaded")
    predicted = predictor.predict(list(df['comment_text']))
    print("predicted")

    results = list(map(lambda pred: float(np.argmax(pred['logits'])), predicted))
    return results, df


def predict_test():
    results, df = predict(
        archive=ARCHIVE_PATH,
        file=FILE_PATH,
    )
    print("results created")
    result_df = pd.DataFrame(
        data={
            'id': df['id'],
            'prediction': results
        },
        columns=('id', 'prediction')
    )
    print("result_df created")
    result_df.to_csv('../dataset/result.csv', index=False)


if __name__ == '__main__':
    predict_test()
