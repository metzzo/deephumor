import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from xgboost import XGBClassifier

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

    clf = XGBClassifier(
        learning_rate=0.1,
        n_estimators=500,
        max_depth=5,
        min_child_weight=3,
        gamma=0.2,
        subsample=0.6,
        colsample_bytree=1.0,
        nthread=8,
        scale_pos_weight=1,
        seed=27,
        tree_method='gpu_hist',
        verbose=True,
    )

    clf.fit(X=train_punchline, y=train_category)
    pred = clf.predict(validation_punchline)
    print("", accuracy_score(
        y_true=validation_category,
        y_pred=pred
    ))




if __name__ == '__main__':
    main()