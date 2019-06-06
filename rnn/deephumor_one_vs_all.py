from typing import Dict

import numpy as np
import torch
import torch.optim as optim

from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import Elmo
from allennlp.modules.elmo import batch_to_ids
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper, CnnEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, ElmoTokenEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.training.trainer import Trainer
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from rnn import DeepHumorDatasetReader, MeanAbsoluteError

EMBEDDING_DIM = 256
HIDDEN_DIM = 128
BATCH_SIZE = 64

# Model in AllenNLP represents a model that is trained.
@Model.register("lstm_classifier")
class LstmClassifier(Model):
    def __init__(self,
                 reader: DeepHumorDatasetReader,
                 weight: float,
                 vocab) -> None:
        super().__init__(None)
        self.decision = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(16),
        ).cuda()

        self.final_decision = torch.nn.Sequential(
            torch.nn.Linear(16, 1),
        ).cuda()

        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(1)

        self.loss_function = BCEWithLogitsLoss(
            pos_weight=torch.tensor(weight)
        )

        self.threshold = torch.tensor([0.5] * BATCH_SIZE).cuda()

    def forward(self,
                meaning,
                label: torch.Tensor = None) -> torch.Tensor:
        logits = self.final_decision(self.decision(meaning))

        output = {"logits": logits}
        if label is not None:
            activation = torch.sigmoid(logits)

            if activation.size(0) == BATCH_SIZE:
                thres = self.threshold
            else:
                thres = torch.tensor([0.5] * BATCH_SIZE).cuda()

            actual_predicted = torch.gt(activation, thres).long()
            predicted = torch.FloatTensor(activation.size(0), 2).cuda()
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
            torch.nn.Linear(7 * 16, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=64,
                out_features=7
            )
        ).cuda()

        self.accuracy = CategoricalAccuracy()
        self.f1_measures = [F1Measure(positive_label=i) for i in range(0, 7)]
        self.mae = MeanAbsoluteError()


    # Instances are fed to forward after batching.
    # Fields are passed through arguments with the same name.
    def forward(self,
                meaning,
                label: torch.Tensor = None) -> torch.Tensor:
        def apply_model(model, idx):
            result = model(meaning, (label == idx).int())['logits'].flatten()
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


def get_one_vs_all_model(positive_label, vocab):
    best_performance = None
    best_weight = None
    best_model = None
    for weight in [2.5]:#[1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0]:
        print("Train one vs all for label ", positive_label)
        reader = DeepHumorDatasetReader(positive_label=positive_label)

        train_dataset = reader.read('train')
        dev_dataset = reader.read('val')

        model = LstmClassifier(reader, weight, vocab).cuda()

        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)

        iterator = BasicIterator(batch_size=BATCH_SIZE)

        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=train_dataset,
                          validation_dataset=dev_dataset,
                          patience=25,
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
    return best_model

def get_final_classifier(models, vocab):
    print("Train final classifier ")
    reader = DeepHumorDatasetReader()

    train_dataset = reader.read('train')
    dev_dataset = reader.read('val')

    final_model = FinalClassifier(vocab=vocab, models=models)

    optimizer = optim.Adam(final_model.parameters(), lr=1e-5, weight_decay=1e-5)

    iterator = BasicIterator(batch_size=BATCH_SIZE)

    trainer = Trainer(model=final_model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      patience=25,
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