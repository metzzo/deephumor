import datetime
import os
import string
import time
import copy

import torch
from torch import nn, autograd
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from datamanagement.cartoon_lstm_dataset import n_letters
from evaluation.accuracy_evaluation import AccuracyEvaluation
from evaluation.loss_evaluation import LossEvaluation
from evaluation.overall_evaluation import OverallEvaluation

n_hidden = 128
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


class Network(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_size):
        super(Network, self).__init__()

        vocab_size = n_letters

        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to classification space
        self.hidden2classification = nn.Linear(hidden_dim, output_size)
        self.dropout_layer = nn.Dropout(p=0.2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, batch, lengths):
        self.hidden = self.init_hidden(batch.size(-1))

        embeds = self.embedding(batch)
        packed_input = pack_padded_sequence(embeds, lengths)
        outputs, (ht, ct) = self.lstm(packed_input, self.hidden)

        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = self.softmax(output)

        return output

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
                autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))


def train_lstm_model(
        criterion,
        optimizer,
        scheduler,
        training_dataset,
        validation_dataset,
        batch_size,
        device,
        num_epochs=25):
    assert batch_size == 1 # only supported yet

    network = Network(
        input_size=n_letters,
        hidden_size=n_hidden,
        output_size=7,
    )
    network.to(device)

    optimizer = optimizer(params=network.parameters())
    scheduler = scheduler(optimizer=optimizer)
    target = training_dataset.df['funniness']
    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = len(target) / class_sample_count
    samples_weight = np.array([weight[t - 1] for t in target])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    dataloaders = {
        "train": DataLoader(
            dataset=training_dataset,
            batch_size=batch_size,
            num_workers=0,
            sampler=sampler
        ),
        "val": DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
    }

    evaluations = {
        "train": OverallEvaluation(num=len(dataloaders["train"]), batch_size=batch_size).add_evaluations(
            [LossEvaluation, AccuracyEvaluation]
        ),
        "val": OverallEvaluation(num=len(dataloaders["val"]), batch_size=batch_size).add_evaluations(
            [AccuracyEvaluation]
        ),
    }

    target_path = os.path.abspath("./../models/{0}_lstm_model.pth".format(
        str(datetime.datetime.today()).replace('-', '').replace(':', '').replace(' ', '_')
    ))
    since = time.time()

    best_network_wts = copy.deepcopy(network.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            evaluations[phase].reset()

            if phase == 'train':
                scheduler.step()
                network.train()  # Set network to training mode
            else:
                network.eval()   # Set network to evaluate mode

            # Iterate over data.
            data = list(dataloaders[phase])
            for entry in data:
                pass



            for _, inputs, labels in data: # [:100]:
                inputs = inputs.to(device)
                labels = labels.to(device).float()

                input = inputs[0].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    hidden = network.init_hidden().to(device)
                    for i in range(input.size()[0]):
                        output, hidden = network(input[i], hidden)

                    loss = criterion(output, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # Get top N categories
                    _, topi = output.data.topk(1, 1, True)

                # statistics
                evaluations[phase].add_entry(predictions=topi.flatten(), actual_label=labels, loss=loss.item())

            print('{0} evaluation:\n {1}'.format(
                phase, str(evaluations[phase])))

        # deep copy the network
        epoch_acc = evaluations['val'].accuracy_evaluation.accuracy
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_network_wts = copy.deepcopy(network.state_dict())
            torch.save(network.state_dict(), target_path)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    network.load_state_dict(best_network_wts)

    return network
