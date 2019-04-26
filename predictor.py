import numpy as np
from allennlp.models import load_archive

from jigsaw import *

predictor = Predictor.from_archive(
    load_archive("./result_2/model.tar.gz", cuda_device=0),
    predictor_name='sentence_classifier_predictor'
)
print("Predictor loaded")
df = pd.read_csv('../dataset/jigsaw/test.csv')
print("CSV loaded")
predicted = predictor.predict(list(df['comment_text']))
print("predicted")

results = list(map(lambda pred: float(np.argmax(pred['logits'])), predicted))
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
