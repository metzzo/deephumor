import pandas as pd
import os
import sklearn

import argparse

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
    ('funniness_rfischer', 'funniness_eidenberger')
]:
    mae = mean_absolute_error(df[p1].values, df[p2].values)
    accuracy = accuracy_score(df[p1].values, df[p2].values)
    print(p1, p2, mae, accuracy)