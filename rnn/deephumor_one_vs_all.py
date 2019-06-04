from typing import Dict

import numpy as np
import torch
import torch.optim as optim

from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper, CnnEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, ElmoTokenEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.training.trainer import Trainer
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from jigsaw import DeepHumorDatasetReader, SentenceClassifierPredictor, WassersteinLoss, one_hot_embedding, \
    MeanAbsoluteError

EMBEDDING_DIM = 256
HIDDEN_DIM = 128
BATCH_SIZE = 64

# Model in AllenNLP represents a model that is trained.
@Model.register("lstm_classifier")
class LstmClassifier(Model):
    def __init__(self,
                 reader: DeepHumorDatasetReader,
                 weight: float,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)

        self.word_embeddings = word_embeddings

        self.encoder = encoder

        self.decision = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(encoder.get_output_dim(), 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1),
        ).cuda()

        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(1)

        #distr = reader.distribution[self.]

        #self.loss_function = WassersteinLoss(size=2)
        pos_weight = reader.distribution[reader.positive_label]
        print("pos weight", pos_weight)
        self.loss_function = BCEWithLogitsLoss(
            pos_weight=torch.tensor(weight)
        )

        self.threshold = torch.tensor([0.5] * BATCH_SIZE).cuda()

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)

        # Forward pass
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.decision(encoder_out)


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

        self.decision = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(7),
            torch.nn.Linear(7, 7),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(7),
            torch.nn.Linear(7, 7),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(7),
            torch.nn.Linear(
                in_features=7,
                out_features=vocab.get_vocab_size('labels')
            )
        ).cuda()

        self.accuracy = CategoricalAccuracy()
        self.f1_measures = [F1Measure(positive_label=i) for i in range(0, 7)]
        self.mae = MeanAbsoluteError()


    # Instances are fed to forward after batching.
    # Fields are passed through arguments with the same name.
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> torch.Tensor:
        def apply_model(model, idx):
            result = model(tokens, (label == idx).int())['logits'].flatten()
            return result


        results = [apply_model(model, idx) for idx, model in enumerate(self.models)]
        combined = torch.stack(results, 1)
        logits = self.decision(combined)

        # In AllenNLP, the output of forward() is a dictionary.
        # Your output dictionary must contain a "loss" key for your model to be trained.
        output = {"logits": logits}
        if label is not None:
            _, predicted = torch.max(logits, 1)

            self.accuracy(logits, label)
            for f1 in self.f1_measures:
                f1(logits, label)

            output["loss"] = self.loss_function(logits, label)
            self.mae(predicted, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        results = {
            'accuracy': self.accuracy.get_metric(reset),
            'mae': self.mae.get_metric(reset),
        }
        for index, f1 in enumerate(self.f1_measures):
            precision, recall, f1_measure = f1.get_metric(reset)
            results.update({
                #'{}_precision'.format(index + 1): precision,
                #'{}_recall'.format(index + 1): recall,
                '{}_f1_measure'.format(index + 1): f1_measure
            })
        return results


def     get_one_vs_all_model(positive_label, vocab):
    best_model = None
    best_performance = None
    best_weight = None

    encoder = PytorchSeq2VecWrapper(
        torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True)
    )

    # original
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    # small
    options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
                    '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json')
    weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
                   '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')

    elmo_embedder = ElmoTokenEmbedder(
        options_file,
        weight_file,
        dropout=0.5,
        do_layer_norm=False,
    ).cuda()

    # token_embedding = Embedding(
    #    num_embeddings=vocab.get_vocab_size('tokens'),
    #    embedding_dim=EMBEDDING_DIM
    # ).cuda()

    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder}).cuda()

    # encoder = CnnEncoder(
    #    embedding_dim=EMBEDDING_DIM,
    #    num_filters=6,
    # )

    for weight in [3.0]:#[1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0]:
        print("Train one vs all for label ", positive_label)
        reader = DeepHumorDatasetReader(positive_label=positive_label)

        train_dataset = reader.read('C:/Users/rfischer/Development/DeepHumor/export/original_export/train_set.p')
        dev_dataset = reader.read('C:/Users/rfischer/Development/DeepHumor/export/original_export/validation_set.p')

        model = LstmClassifier(reader, weight, word_embeddings, encoder, vocab).cuda()

        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

        iterator = BasicIterator(batch_size=BATCH_SIZE, sorting_keys=[("tokens", "num_tokens")])

        iterator.index_with(vocab)

        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=train_dataset,
                          validation_dataset=dev_dataset,
                          patience=75,
                          cuda_device=0,
                          validation_metric='+f1_measure',
                          num_epochs=1000)


        results = trainer.train()
        performance = results['best_validation_f1_measure']
        if best_performance is None or best_performance < performance:
            best_performance = performance
            best_model = model
            best_weight = weight
    print("Result for ", positive_label, str(results))
    print("With weight ", best_weight)
    raise NotImplementedError()
    return best_model

def get_final_classifier(models, vocab):
    print("Train final classifier ")
    reader = DeepHumorDatasetReader()

    train_dataset = reader.read('C:/Users/rfischer/Development/DeepHumor/export/original_export/train_set.p')
    dev_dataset = reader.read('C:/Users/rfischer/Development/DeepHumor/export/original_export/validation_set.p')

    final_model = FinalClassifier(vocab=vocab, models=models)

    optimizer = optim.Adam(final_model.parameters(), lr=1e-5, weight_decay=1e-5)

    iterator = BucketIterator(batch_size=BATCH_SIZE, sorting_keys=[("tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=final_model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      patience=30,
                      cuda_device=0,
                      num_epochs=500)

    results = trainer.train()
    print("Final result", str(results))

    return final_model

def main():
    # apply models and train final classiier

    reader = DeepHumorDatasetReader()

    train_dataset = reader.read('C:/Users/rfischer/Development/DeepHumor/export/original_export/train_set.p')
    dev_dataset = reader.read('C:/Users/rfischer/Development/DeepHumor/export/original_export/validation_set.p')

    vocab = Vocabulary.from_instances(train_dataset + dev_dataset, min_count={'tokens': 3})

    models = list([get_one_vs_all_model(i, vocab) for i in range(0, 7)])
    final_model = get_final_classifier(models=models, vocab=vocab)




if __name__ == '__main__':
    main()