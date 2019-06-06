import os
import pickle

import nltk
import torch
from typing import Iterator, List, Dict

import pandas as pd
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField, MetadataField, ArrayField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
import numpy as np
from allennlp.modules import Elmo

# small
from allennlp.modules.elmo import batch_to_ids
from sklearn.preprocessing import StandardScaler

options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
                '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json')
weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
               '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')

TRAIN_PATH = "/home/rfischer/Documents/DeepHumor/train_set.p"
VALIDATION_PATH = "/home/rfischer/Documents/DeepHumor/validation_set.p"


def get_df_distribution(df):
    groups = df.groupby(by='funniness').groups
    counts = {group: len(groups[group]) for group in groups}
    counts = np.array([counts[group] for group in sorted(groups.keys())])

    total_count = len(df)

    distribution = counts / float(total_count)

    return counts, total_count, distribution


@DatasetReader.register('deephumor-dataset')
class DeepHumorDatasetReader(DatasetReader):
    def __init__(self, lazy=False, positive_label=None) -> None:
        super().__init__(lazy=lazy)

        self.train_df = pickle.load(open(TRAIN_PATH, "rb"))
        self.validation_df = pickle.load(open(VALIDATION_PATH, "rb"))

        train_punchlines = self.train_df['punchline'].apply(nltk.word_tokenize)
        validation_punchlines = self.validation_df['punchline'].apply(nltk.word_tokenize)

        if not os.path.exists('word_embedding.p'):
            print("get elmo vectors")
            elmo = Elmo(options_file, weight_file, 2, dropout=0)
            character_ids = batch_to_ids(list(train_punchlines) + list(validation_punchlines))
            embeddings = elmo(character_ids)
            self.feature_vectors = torch.cat(embeddings['elmo_representations'], dim=2)
            self.feature_vectors = self.feature_vectors.mean(dim=1)
            pickle.dump(self.feature_vectors, open('word_embedding.p', "wb"), protocol=4)
        else:
            self.feature_vectors = pickle.load(open('word_embedding.p', "rb"))

        scaler = StandardScaler()

        self.train_feature_vec = self.feature_vectors[:len(self.train_df), :]
        self.train_feature_vec = self.train_feature_vec.detach().cpu().numpy()
        self.train_feature_vec = scaler.fit_transform(self.train_feature_vec)
        self.train_label_vec = (np.array(self.train_df[['funniness']]).flatten() - 1).astype(int)

        self.validation_feature_vec = self.feature_vectors[len(self.train_df):, :]
        self.validation_feature_vec = self.validation_feature_vec.detach().cpu().numpy()
        self.validation_feature_vec = scaler.transform(self.validation_feature_vec)
        self.validation_label_vec = (np.array(self.validation_df[['funniness']]).flatten() - 1).astype(int)

        self.positive_label = positive_label
        if self.positive_label is not None:
            func = np.vectorize(lambda x: 1 if x == self.positive_label else 0)
            self.train_label_vec = func(self.train_label_vec)
            self.validation_label_vec = func(self.validation_label_vec)

        # oversampling the train df

        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=0)
        self.train_feature_vec, self.train_label_vec = ros.fit_resample(self.train_feature_vec, self.train_label_vec)

    def text_to_instance(self, feature, target=None) -> Instance:
        meaning_field = ArrayField(feature)
        fields = {
            "meaning": meaning_field,
        }

        if target is not None:
            target_field = LabelField(target, skip_indexing=True)
            fields["label"] = target_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        if file_path == 'train':
            feature_vec = self.train_feature_vec
            label_vec = self.train_label_vec
        else:
            feature_vec = self.validation_feature_vec
            label_vec = self.validation_label_vec

        for i in range(0, len(feature_vec)):
            funniness = int(label_vec[i])
            feature = feature_vec[i]

            yield self.text_to_instance(feature, funniness)
