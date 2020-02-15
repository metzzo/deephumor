import pandas as pd
import os
import sklearn

import argparse

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('base_path',
                    type=str,
                    help='The base path')

args = parser.parse_args()

rfischer_path = os.path.join(args.base_path, 'rfischer_annotations.csv')
eidenberger_path = os.path.join(args.base_path, 'eidenberger_annotations.csv')
admin_path = os.path.join(args.base_path, 'admin_annotations.csv')


rfischer_df = pd.read_csv(rfischer_path, sep=";")
eidenberger_df = pd.read_csv(eidenberger_path, sep=";")
admin_df = pd.read_csv(admin_path, sep=";")

merged = pd.merge(
    pd.merge(
        rfischer_df,
        eidenberger_df,
        suffixes=('_rfischer', '_eidenberger'),
        on='cartoon_id'),
    admin_df,
    on='cartoon_id',
    suffixes=('', '_admin')
)
df = merged[['funniness_rfischer', 'funniness_eidenberger', 'funniness']]
df = df.dropna()


for p1, p2 in [
    ('funniness', 'funniness_rfischer'),
    ('funniness', 'funniness_eidenberger'),
    ('funniness_rfischer', 'funniness_eidenberger'),
    ('funniness_rfischer', 'dummy_classifier'),
    ('funniness_eidenberger', 'dummy_classifier'),
    ('funniness_rfischer', 'dummy_regressor'),
    ('funniness_eidenberger', 'dummy_regressor')
]:
    rounded_d2 = None
    d1 = df[p1].values
    if p2 == 'dummy_classifier':
        clf = DummyClassifier(strategy='most_frequent')
        clf.fit(df, d1)
        d2 = clf.predict(df)
    elif p2 == 'dummy_regressor':
        clf = DummyRegressor(strategy='mean')
        clf.fit(df, d1)
        d2 = clf.predict(df)
        rounded_d2 = (d2 + 0.5).astype(int)
    else:
        d2 = df[p2].values

    mae = mean_absolute_error(d1, d2)
    accuracy = accuracy_score(d1, d2 if rounded_d2 is None else rounded_d2)


    print(p1, p2, mae, accuracy)