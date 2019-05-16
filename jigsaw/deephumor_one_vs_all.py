from typing import Dict

import numpy as np
import torch
import torch.optim as optim

from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.training.trainer import Trainer
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from jigsaw import DeepHumorDatasetReader, SentenceClassifierPredictor, WassersteinLoss, one_hot_embedding

EMBEDDING_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 32

# Model in AllenNLP represents a model that is trained.
@Model.register("lstm_classifier")
class LstmClassifier(Model):
    def __init__(self,
                 reader: DeepHumorDatasetReader,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)

        self.word_embeddings = word_embeddings

        self.encoder = encoder

        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=2)

        self.linear = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(encoder.get_output_dim(), 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(1)

        #distr = reader.distribution[self.]

        #self.loss_function = WassersteinLoss(size=2)
        pos_weight = reader.distribution[reader.positive_label]
        print("pos weight", pos_weight)
        self.loss_function = BCEWithLogitsLoss(
            pos_weight=torch.tensor(3.0)
        )

        self.threshold = torch.tensor([0.5] * BATCH_SIZE).cuda()

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)

        # Forward pass
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)


        output = {"logits": logits}
        if label is not None:
            activation = torch.sigmoid(logits)

            if activation.size(0) == BATCH_SIZE:
                thres = self.threshold
            else:
                thres = torch.tensor([0.5] * BATCH_SIZE).cuda()

            actual_predicted = torch.gt(activation, thres).long()
            predicted = torch.FloatTensor(activation.size(0), 2).cuda()

            # In your for loop
            predicted.zero_()
            predicted.scatter_(1, actual_predicted, 1)

            self.accuracy(predicted, label)
            self.f1_measure(predicted, label)
            output["loss"] = self.loss_function(logits.flatten(), label.float())

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1_measure = self.f1_measure.get_metric(reset)
        return {'accuracy': self.accuracy.get_metric(reset),
                'precision': precision,
                'recall': recall,
                'f1_measure': f1_measure}


class FinalClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 models) -> None:
        super().__init__(vocab)
        self.models = models

        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        self.loss_function = torch.nn.CrossEntropyLoss()

        self.linear = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=vocab.get_vocab_size('labels')
        )


    # Instances are fed to forward after batching.
    # Fields are passed through arguments with the same name.
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> torch.Tensor:

        results = [model(tokens, label) for model in self.models]
        combined = torch.cat(list(results), 1)

        # In AllenNLP, the output of forward() is a dictionary.
        # Your output dictionary must contain a "loss" key for your model to be trained.
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            self.f1_measure(logits, label)
            output["loss"] = self.loss_function(logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1_measure = self.f1_measure.get_metric(reset)
        return {'accuracy': self.accuracy.get_metric(reset),
                'precision': precision,
                'recall': recall,
                'f1_measure': f1_measure}


def get_one_vs_all_model(positive_label, word_embeddings, encoder, vocab):
    print("Train one vs all for label ", positive_label)
    reader = DeepHumorDatasetReader(positive_label=positive_label)

    train_dataset = reader.read('C:/Users/rfischer/Development/DeepHumor/export/original_export/train_set.p')
    dev_dataset = reader.read('C:/Users/rfischer/Development/DeepHumor/export/original_export/validation_set.p')

    model = LstmClassifier(reader, word_embeddings, encoder, vocab).cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    iterator = BucketIterator(batch_size=BATCH_SIZE, sorting_keys=[("tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      patience=10,
                      cuda_device=0,
                      num_epochs=20)

    trainer.train()

    return model

def get_final_classifier(models, word_embeddings, encoder, vocab):
    print("Train final classifier ")
    reader = DeepHumorDatasetReader()

    train_dataset = reader.read('C:/Users/rfischer/Development/DeepHumor/export/original_export/train_set.p')
    dev_dataset = reader.read('C:/Users/rfischer/Development/DeepHumor/export/original_export/validation_set.p')

    final_model = FinalClassifier(vocab=vocab, models=models)

    optimizer = optim.Adam(final_model.parameters(), lr=1e-4, weight_decay=1e-5)

    iterator = BucketIterator(batch_size=BATCH_SIZE, sorting_keys=[("tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=final_model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      patience=10,
                      cuda_device=0,
                      num_epochs=20)

    trainer.train()

    return final_model

def main():
    # apply models and train final classiier

    reader = DeepHumorDatasetReader()

    train_dataset = reader.read('C:/Users/rfischer/Development/DeepHumor/export/original_export/train_set.p')
    dev_dataset = reader.read('C:/Users/rfischer/Development/DeepHumor/export/original_export/validation_set.p')

    vocab = Vocabulary.from_instances(train_dataset + dev_dataset, min_count={'tokens': 3})

    token_embedding = Embedding(
        num_embeddings=vocab.get_vocab_size('tokens'),
        embedding_dim=EMBEDDING_DIM
    ).cuda()

    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding}).cuda()

    encoder = PytorchSeq2VecWrapper(
        torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True).cuda(),
    )

    models = list([get_one_vs_all_model(i, word_embeddings, encoder, vocab) for i in range(0, 7)])
    final_model = get_final_classifier(models=models, word_embeddings=word_embeddings, encoder=encoder, vocab=vocab)




if __name__ == '__main__':
    main()