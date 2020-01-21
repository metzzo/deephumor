import datetime
import pickle
import time

import nltk
import pandas as pd
import numpy as np

from hpsklearn import HyperoptEstimator, svc, any_classifier, any_preprocessing
from sklearn.decomposition import PCA

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, mean_absolute_error
from allennlp.modules.elmo import Elmo, batch_to_ids

import torch
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = "/home/rfischer/Documents/DeepHumor/train_set.p"
VALIDATION_PATH = "/home/rfischer/Documents/DeepHumor/test_set.p"

#options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
#weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)

def main():
    train_df = pickle.load(open(TRAIN_PATH, "rb"))
    validation_df = pickle.load(open(VALIDATION_PATH, "rb"))

    train_punchlines = train_df['punchline'].apply(nltk.word_tokenize)
    validation_punchlines = validation_df['punchline'].apply(nltk.word_tokenize)

    print("get elmo vectors")
    elmo = Elmo(options_file, weight_file, 2, dropout=0)
    character_ids = batch_to_ids(list(train_punchlines) + list(validation_punchlines))
    embeddings = elmo(character_ids)
    feature_vectors = torch.cat(embeddings['elmo_representations'], dim=2)
    feature_vectors = feature_vectors.mean(dim=1)
    #feature_vectors = feature_vectors.reshape(feature_vectors.size(0), -1)

    scaler = StandardScaler()

    train_feature_vec = feature_vectors[:len(train_punchlines), :]
    train_feature_vec = train_feature_vec.detach().cpu().numpy()
    train_feature_vec = scaler.fit_transform(train_feature_vec)

    validation_feature_vec = feature_vectors[len(train_punchlines):, :]
    validation_feature_vec = validation_feature_vec.detach().cpu().numpy()
    validation_feature_vec = scaler.transform(validation_feature_vec)

    train_punchline = train_feature_vec
    train_category = np.array(train_df[['funniness']]).flatten()
    print("fit & transform PCA")
    #pca = PCA(n_components=500)
    #train_punchline = pca.fit_transform(train_punchline, train_category)

    validation_punchline = validation_feature_vec #pca.transform(validation_feature_vec)
    validation_category = np.array(validation_df[['funniness']]).flatten()
    """


    vectorizer = TfidfVectorizer()
    train_punchline = vectorizer.fit_transform(train_df['punchline']).toarray()
    train_category = np.array(train_df[['funniness']]).flatten()

    validation_punchline = vectorizer.transform(validation_df['punchline']).toarray()
    validation_category = np.array(validation_df[['funniness']]).flatten()
    """

    import time

    start = time.time()
    clf = HyperoptEstimator(
        classifier=svc('my_clf'),
        max_evals=50,
        trial_timeout=250,
        seed=42,
    )
    clf.fit(X=train_punchline, y=train_category)
    end = time.time()
    print("Train duration", end - start)

    print(clf.score(validation_punchline, validation_category))

    clf = pickle.load( open('tfidf/automl_tfidf.p', 'rb'))

    pred = clf.predict(validation_punchline)
    print("Accuracy ", accuracy_score(
        y_true=validation_category,
        y_pred=pred
    ))

    print("MAE ", mean_absolute_error(
        y_true=validation_category,
        y_pred=pred
    ))

    pickle.dump(clf, open('tfidf/automl_result_{}'.format(datetime.datetime.now()), "wb"), protocol=4)


if __name__ == '__main__':
    main()