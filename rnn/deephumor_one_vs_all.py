import datetime
import pickle
from itertools import combinations
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
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from rnn import DeepHumorDatasetReader, MeanAbsoluteError, VALIDATION_PATH, TRAIN_PATH

BATCH_SIZE = 64


def init_weights(m):
    if type(m) == nn.Linear:
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

def subsets(s):
    for cardinality in range(len(s) + 1):
        yield from combinations(s, cardinality)


class OneVRestClassifier(Model):
    def __init__(self,
                 reader: DeepHumorDatasetReader,
                 weight: float) -> None:
        super().__init__(None)
        self.decision = torch.nn.Sequential(
            torch.nn.Linear(4262, 16),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(16),
        ).cuda()

        self.final_decision = torch.nn.Sequential(
            torch.nn.Linear(16, 1),
            torch.nn.ReLU(),
        ).cuda()

        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(1)

        self.loss_function = BCEWithLogitsLoss(
            pos_weight=torch.tensor(weight)
        )
        self.threshold = torch.tensor([0.5] * BATCH_SIZE).cuda()


    def forward(self,
                spacy_meaning,
                tfidf_meaning,
                label: torch.Tensor = None) -> torch.Tensor:
        meaning = torch.cat([spacy_meaning, tfidf_meaning], dim=1)
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
                 models,
                 classes) -> None:
        super().__init__(None)
        self.models = models
        self.classes = classes

        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        self.loss_function = torch.nn.L1Loss()

        self.decision = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(len(self.classes) * 8, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(
                in_features=32,
                out_features=1
            ),
            torch.nn.ReLU()
        ).cuda()

        self.mae = MeanAbsoluteError()

        self.decision.apply(init_weights)
        self.prepare = {}
        for model in models:
            self.prepare[model] = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(16, 8),
                torch.nn.BatchNorm1d(8),
                torch.nn.ReLU(),
            ).cuda()
            self.prepare[model].apply(init_weights)


    # Instances are fed to forward after batching.
    # Fields are passed through arguments with the same name.
    def forward(self,
                spacy_meaning,
                tfidf_meaning,
                label: torch.Tensor = None) -> torch.Tensor:

        meaning = torch.cat([spacy_meaning, tfidf_meaning], dim=1)

        def apply_model(model):
            result = model.decision(meaning)
            result = self.prepare[model](result)
            return result

        results = [apply_model(model) for model in self.models]
        combined = torch.stack(results, 1)
        combined = combined.reshape(combined.size(0), -1)
        logits = self.decision(combined)

        # In AllenNLP, the output of forward() is a dictionary.
        # Your output dictionary must contain a "loss" key for your model to be trained.
        output = {"logits": logits}
        if label is not None:
            predicted = (logits * 6).float()
            label = label.reshape(-1, 1).float()

            output["loss"] = self.loss_function(logits, label / 6.0)
            self.mae(predicted, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        results = {
            'mae': self.mae.get_metric(reset),
        }
        return results


def get_one_vs_all_model(positive_labels):
    best_performance = None
    best_weight = None
    best_model = None
    str_labels = ' '.format(positive_labels)
    for weight in [2.5]:#[1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0]:
        print("Train one vs all for label ", str(str_labels))
        reader = DeepHumorDatasetReader(positive_labels=positive_labels)

        train_dataset = reader.read('train')
        dev_dataset = reader.read('val')

        model = OneVRestClassifier(reader, weight).cuda()

        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)

        iterator = BasicIterator(batch_size=BATCH_SIZE)

        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=train_dataset,
                          validation_dataset=dev_dataset,
                          patience=100,
                          cuda_device=0,
                          validation_metric='+f1_measure',
                          num_epochs=10000)


        results = trainer.train()
        performance = results['best_validation_f1_measure']
        if best_performance is None or best_performance < performance:
            best_performance = performance
            best_model = model
            best_weight = weight
    print("Result for ", str_labels, str(results))
    print("With weight ", best_weight)
    return best_model

def get_final_classifier(models, classes):
    print("Train final classifier ")
    reader = DeepHumorDatasetReader()

    train_dataset = reader.read('train')
    dev_dataset = reader.read('val')

    final_model = FinalClassifier(models=models, classes=classes)

    optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=0.001)

    iterator = BasicIterator(batch_size=256)

    trainer = Trainer(model=final_model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      patience=25,
                      cuda_device=0,
                      num_epochs=10000)

    results = trainer.train()
    print("Final result", str(results))

    return final_model

def get_dummy_performance():
    validation_df = pickle.load(open(VALIDATION_PATH, "rb"))
    train_df = pickle.load(open(TRAIN_PATH, "rb"))

    regressor = DummyRegressor()
    regressor.fit(None, train_df['funniness'])
    predicted = regressor.predict(validation_df['punchline'])
    print("Dummy MAE", mean_absolute_error(validation_df['funniness'], predicted))

def main():
    # apply models and train final classiier
    #classes = [6, 5, 4, 3, 2, 1, 0]
    #subset_classes = list(filter(lambda x: len(x) < len(classes) and len(x) > 1, subsets(classes)))
    subset_classes = [
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [5, 6],
    ]
    """
    
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
    """
    #models = list([get_one_vs_all_model(cl) for cl in subset_classes])
    #torch.save(models, open('models.p', "wb"))
    models = torch.load(open('models.p', "rb"))

    final_model = get_final_classifier(models=models, classes=subset_classes)

    torch.save({
        "final_model": final_model,
        "models": models,
    }, "model_{}.pth".format(str(datetime.datetime.now())))




if __name__ == '__main__':
    get_dummy_performance()
    main()
