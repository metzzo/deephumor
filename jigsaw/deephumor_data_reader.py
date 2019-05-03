import pickle
from typing import Iterator, List, Dict

import pandas as pd
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token


@DatasetReader.register('deephumor-dataset')
class DeepHumorDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, lazy=False) -> None:
        super().__init__(lazy=lazy)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

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
        df = pickle.load(open(file_path, "rb"))
        for _, data in df.iterrows():
            target = str((int(data['funniness']) - 1))
            punchline = data['punchline'].strip().split() #{} {}'.format(target, data['punchline'].strip().split())
            yield self.text_to_instance([Token(word) for word in punchline], target)
