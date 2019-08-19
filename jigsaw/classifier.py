from typing import Dict

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2VecEncoder, Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, MeanAbsoluteError


@Model.register('jigsaw-lstm')
class LstmClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings

        self.encoder = encoder

        self.convs = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.BatchNorm2d(1),

            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.AvgPool2d(64),
        )

        self.linear = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(24, vocab.get_vocab_size('labels'))
        )

        self.accuracy = CategoricalAccuracy()
        #self.f1_measure = F1Measure(positive_label=1)

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.mae = MeanAbsoluteError()

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None):
        mask = get_text_field_mask(tokens)

        embeddings = self.word_embeddings(tokens)
        x = self.encoder(embeddings, mask)

        #x = encoder_out.reshape(encoder_out.shape[0], 1, 32, 32)
        #x = self.convs(x)
        x = x.reshape(x.size(0), -1)

        #additional_data = torch.tensor([[0.0]] * x.shape[0]).cuda()
        #x = torch.cat((x, additional_data), dim=1)
        x = self.linear(x)

        logits = x
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            #self.f1_measure(logits, label)
            output["loss"] = self.loss_function(logits, label)
            _, predicted = torch.max(logits, 1)
            self.mae(predicted, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        #precision, recall, f1_measure = self.f1_measure.get_metric(reset)
        return {
            'accuracy': self.accuracy.get_metric(reset),
            'mae': self.mae.get_metric(reset),
            #'precision': precision,
            #'recall': recall,
            #'f1_measure': f1_measure
        }

