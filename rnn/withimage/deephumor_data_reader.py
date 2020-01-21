import os
import pickle

from scipy import stats

import spacy
import torch
from typing import Iterator, List, Dict

import pandas as pd
from PIL import Image, ImageOps
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField, MetadataField, ArrayField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
import numpy as np
from allennlp.modules import Elmo

# small
from allennlp.modules.elmo import batch_to_ids
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

nlp = spacy.load('en_core_web_lg')

options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
                '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json')
weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
               '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')

TRAIN_PATH = "/home/rfischer/Documents/DeepHumor/train_set.p"
VALIDATION_PATH = "/home/rfischer/Documents/DeepHumor/validation_set.p"
TEST_PATH = "/home/rfischer/Documents/DeepHumor/test_set.p"

IMAGES_PATH = '../../../data/original_export/'

def get_df_distribution(df):
    groups = df.groupby(by='funniness').groups
    counts = {group: len(groups[group]) for group in groups}
    counts = np.array([counts[group] for group in sorted(groups.keys())])

    total_count = len(df)

    distribution = counts / float(total_count)

    return counts, total_count, distribution


@DatasetReader.register('deephumor-dataset')
class DeepHumorDatasetReader(DatasetReader):
    def __init__(self, lazy=False, positive_labels=None) -> None:
        super().__init__(lazy=lazy)

        self.train_df = pickle.load(open(TRAIN_PATH, "rb"))
        self.validation_df = pickle.load(open(VALIDATION_PATH, "rb"))
        self.test_df = pickle.load(open(TEST_PATH, "rb"))
        self.transform = transforms.Compose([
            transforms.RandomCrop(size=(256, 256)),
            transforms.Grayscale()
        ])
        if not os.path.exists('word_embedding.p'):
            vocabulary = set()
            elmo = Elmo(options_file, weight_file, 2, dropout=0)

            def elmo_tokenize(x):
                character_ids = batch_to_ids([list(x)])
                embeddings = elmo(character_ids)
                elmo_feature_vectors = torch.cat(embeddings['elmo_representations'], dim=2)
                return elmo_feature_vectors[0].mean(dim=0).detach().cpu().numpy()

            def tokenize(x):
                doc = nlp(x)
                new = []
                for token in doc:
                    if 'NN' in token.tag_:
                        vocabulary.add(token.text)
                        if not token.is_oov:
                            new.append(token.vector)
                new = np.array(new)

                if len(new) > 0:
                    return new.mean(axis=0)
                else:
                    return np.zeros(300)
            vectorizer = TfidfVectorizer(vocabulary=vocabulary)

            raw_punchlines = pd.concat([self.train_df['punchline'], self.validation_df['punchline'], self.test_df['punchline']]).reset_index()
            self.spacy_punchlines = np.vstack(raw_punchlines['punchline'].apply(tokenize).values)
            #self.elmo_punchlines = np.vstack(raw_punchlines['punchline'].apply(elmo_tokenize).values)
            vectorizer.fit(raw_punchlines['punchline'][:len(self.train_df)])
            self.tfidf_punchlines = vectorizer.transform(raw_punchlines['punchline']).toarray()

            self.feature_vectors = np.hstack([
                self.spacy_punchlines,
                self.tfidf_punchlines,
            ]) # self.elmo_punchlines

            pickle.dump(self.feature_vectors, open('word_embedding.p', "wb"), protocol=4)
        else:
            self.feature_vectors = pickle.load(open('word_embedding.p', "rb"))

        self.train_feature_vec = self.feature_vectors[:len(self.train_df), :]
        self.train_feature_vec = np.hstack([
            self.train_feature_vec,
            np.array(range(0, len(self.train_feature_vec))).reshape(-1, 1)
        ])
        self.train_label_vec = (np.array(self.train_df[['funniness']]).flatten() - 1).astype(int)
        self.train_filenames = self.train_df['filename'].values.flatten()

        self.validation_feature_vec = self.feature_vectors[len(self.train_df):len(self.train_df) + len(self.validation_df), :]
        self.validation_feature_vec = np.hstack([
            self.validation_feature_vec,
            np.array(range(0, len(self.validation_feature_vec))).reshape(-1, 1)
        ])
        self.validation_label_vec = (np.array(self.validation_df[['funniness']]).flatten() - 1).astype(int)
        self.validation_filenames = self.validation_df['filename'].values.flatten()

        self.test_feature_vec = self.feature_vectors[len(self.train_df) + len(self.validation_df):, :]
        self.test_feature_vec = np.hstack([
            self.test_feature_vec,
            np.array(range(0, len(self.test_feature_vec))).reshape(-1, 1)
        ])
        self.test_label_vec = (np.array(self.test_df[['funniness']]).flatten() - 1).astype(int)
        self.test_filenames = self.test_df['filename'].values.flatten()

        self.positive_labels = positive_labels
        if self.positive_labels is not None:
            func = np.vectorize(lambda x: 1 if x in self.positive_labels else 0)
            self.train_label_vec = func(self.train_label_vec)
            self.validation_label_vec = func(self.validation_label_vec)
            self.test_label_vec = func(self.test_label_vec)

        # oversampling the train df

        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=0)
        self.train_feature_vec, self.train_label_vec = ros.fit_resample(self.train_feature_vec, self.train_label_vec)
        if positive_labels is not None:
            self.validation_feature_vec, self.validation_label_vec = ros.fit_resample(self.validation_feature_vec, self.validation_label_vec)


    def text_to_instance(self, feature, filename, target=None) -> Instance:
        spacy_meaning_field = ArrayField(feature[:300])
        tfidf_meaning_field = ArrayField(feature[300:])
        image = ArrayField(self.load_image(filename))
        fields = {
            "spacy_meaning": spacy_meaning_field,
            "tfidf_meaning": tfidf_meaning_field,
            "image": image
        }

        if target is not None:
            target_field = LabelField(target, skip_indexing=True)
            fields["label"] = target_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:

        if file_path == 'train':
            feature_vec = self.train_feature_vec
            label_vec = self.train_label_vec
            filenames = self.train_filenames
        elif file_path == 'test':
            feature_vec = self.test_feature_vec
            label_vec = self.test_label_vec
            filenames = self.test_filenames
        else:
            feature_vec = self.validation_feature_vec
            label_vec = self.validation_label_vec
            filenames = self.validation_filenames

        for i in range(0, len(feature_vec)):
            funniness = int(label_vec[i])
            feature = feature_vec[i]
            filename = filenames[int(feature[-1])]
            feature = feature[:-1]

            yield self.text_to_instance(
                feature=feature,
                filename=filename,
                target=funniness,
            )

    def load_image(self, img_name):
        img_name = os.path.join(IMAGES_PATH, img_name)
        image = Image.open(img_name)
        image = ImageOps.fit(image, (300, 300), method=Image.BILINEAR)
        if self.transform is not None:
            image = self.transform(image)
        image = np.array(image)
        image = stats.zscore(image)

        return image
