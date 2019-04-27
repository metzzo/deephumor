import os

import pandas as pd

SOURCE_FOLDER = '../dataset/raw_jigsaw/'
TARGET_FOLDER = '../dataset/jigsaw/'
SMALL = True

df = pd.read_csv(os.path.join(SOURCE_FOLDER, 'train.csv')) # , nrows=100)
df = df.sample(frac=1)

if SMALL:
    df = df[:1000]

split = int(len(df) * 0.8)
train_df = df.iloc[:split]
validation_df = df.iloc[split:]

print("Train shape ", train_df.shape, " Validation shape ",validation_df.shape)

train_df.to_csv(os.path.join(TARGET_FOLDER, 'train.csv'))
validation_df.to_csv(os.path.join(TARGET_FOLDER, 'validation.csv'))

print("Saved to ", TARGET_FOLDER)