import pickle
from typing import Iterator, List, Dict

import pandas as pd
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField, MetadataField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
import numpy as np

def get_df_distribution(df):
    groups = df.groupby(by='funniness').groups
    counts = {group: len(groups[group]) for group in groups}
    counts = np.array([counts[group] for group in sorted(groups.keys())])

    total_count = len(df)

    distribution = counts / float(total_count)

    return counts, total_count, distribution


@DatasetReader.register('deephumor-dataset')
class DeepHumorDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, lazy=False, positive_label=None) -> None:
        super().__init__(lazy=lazy)
        self.df = None
        self.counts, self.total_count, self.distribution = None, None, None
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.positive_label = positive_label

    def text_to_instance(self, tokens: List[Token], target=None) -> Instance:
        tokens_field = TextField(tokens, self.token_indexers)
        fields = {
            "tokens": tokens_field,
        }

        if target is not None:
            target_field = LabelField(target)
            fields["label"] = target_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        self.df = pickle.load(open(file_path, "rb"))
        self.counts, self.total_count, self.distribution = get_df_distribution(self.df)
        print(self.counts, " ", self.total_count)

        for _, data in self.df.iterrows():
            funniness = int(data['funniness']) - 1

            if self.positive_label is not None:
                funniness = 1 if funniness == self.positive_label else 0

            target = str(funniness)
            punchline = data['punchline'].strip().split()
            yield self.text_to_instance([Token(word) for word in punchline], target)
