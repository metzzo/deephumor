from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides


@Predictor.register("sentence_classifier_predictor")
class SentenceClassifierPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)

    def predict(self, sentences: [str]) -> JsonDict:
        result = []
        for i in range(0, len(sentences), 50):
            print("{} of {}".format(i / 50, len(sentences) / 50))
            sentences_batch = sentences[i : min(i + 50, len(sentences))]
            batch = list(map(lambda x: {"sentence": x}, sentences_batch))
            result += self.predict_batch_json(inputs=batch)
        return result

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.split_words(sentence)
        return self._dataset_reader.text_to_instance(tokens)