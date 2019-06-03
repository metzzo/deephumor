import pickle
import pandas as pd
import numpy as np

from hpsklearn import HyperoptEstimator, svc

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

TRAIN_PATH = "/home/rfischer/Documents/DeepHumor/train_set.p"
VALIDATION_PATH = "/home/rfischer/Documents/DeepHumor/validation_set.p"

def main():
    train_df = pickle.load(open(TRAIN_PATH, "rb"))
    validation_df = pickle.load(open(VALIDATION_PATH, "rb"))

    vectorizer = TfidfVectorizer()
    train_punchline = vectorizer.fit_transform(train_df['punchline']).toarray()
    train_category = np.array(train_df[['funniness']]).flatten()

    validation_punchline = vectorizer.transform(validation_df['punchline']).toarray()
    validation_category = np.array(validation_df[['funniness']]).flatten()

    clf = HyperoptEstimator(classifier=svc('mySVC'))
    clf.fit(X=train_punchline, y=train_category)
    print(clf.score(validation_punchline, validation_category))

    pred = clf.predict(validation_punchline)
    print("", accuracy_score(
        y_true=validation_category,
        y_pred=pred
    ))


if __name__ == '__main__':
    main()