from typing import Iterator, List, Dict

import pandas as pd
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token


@DatasetReader.register('jigsaw-dataset')
class JigsawDatasetReader(DatasetReader):
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
        df = pd.read_csv(file_path)
        for _, data in df.iterrows():
            comment_text = data['comment_text'].strip().split()
            target = str(int(data['target'] > 0.5))
            yield self.text_to_instance([Token(word) for word in comment_text], target)
