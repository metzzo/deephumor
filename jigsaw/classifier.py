from typing import Dict

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2VecEncoder, Embedding
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, MeanAbsoluteError
from sklearn.metrics import mean_absolute_error

from jigsaw.wasserstein_loss import WassersteinLossStab

from emd import EMDLoss

import numpy as np
import ot

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

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
            torch.nn.Linear(128, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, vocab.get_vocab_size('labels')),
        )

        self.accuracy = CategoricalAccuracy()
        self.f1_measures = [F1Measure(positive_label=i) for i in range(0, 7)]

        #self.loss_function = torch.nn.CrossEntropyLoss()

        x = np.arange(7, dtype=np.float32)
        M = (x[:, np.newaxis] - x[np.newaxis, :]) ** 2
        M /= M.max()

        self.loss_function = WassersteinLossStab(torch.from_numpy(M)) # EMDLoss()
        self.y_onehot = None

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
        original_logits = x

        logits = torch.abs(x)

        batch_size = x.shape[0]
        logits = logits.double() #logits.reshape(1, batch_size, 7).double()
        output = {"logits": logits}
        if label is not None:
            self.accuracy(original_logits, label)
            for f1 in self.f1_measures:
                f1(original_logits, label)
            one_hot = one_hot_embedding(labels=label, num_classes=7).cuda().double() #.reshape(1, batch_size, 7)
            one_hot.requires_grad = True
            loss = self.loss_function(logits, one_hot)# / batch_size
            output["loss"] = torch.sum(loss)

            _, predicted = torch.max(original_logits, 1)
            self.mae(predicted, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        data = {
            'accuracy': self.accuracy.get_metric(reset),
            'mae': self.mae.get_metric(reset),
        }

        for index, f1 in enumerate(self.f1_measures):
            precision, recall, f1_measure = f1.get_metric(reset)
            data.update({
                #'{}_precision'.format(index + 1): precision,
                #'{}_recall'.format(index + 1): recall,
                '{}_f1_measure'.format(index + 1): f1_measure
            })

        return data

