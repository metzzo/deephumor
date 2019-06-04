import argparse
from functools import partial

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, L1Loss
from torch.nn.functional import softmax

from cnn_experiments.factory import get_model
from datamanagement.cartoon_lstm_dataset import CartoonLSTMDataset
from datamanagement.factory import get_subset
from models.train_cnn import train_cnn_model
from torch.optim import lr_scheduler

from models.train_lstm import train_lstm_model


def pipeline(source, model, epochs, batch_size, learning_rate, loss, optimizer, device):
    from datamanagement.subset import Subset
    from datamanagement.cartoon_cnn_dataset import CartoonCNNDataset

    torch.manual_seed(42)

    models = [

    ]
    losses = {
        'nll': nn.NLLLoss,
        'l1': nn.L1Loss,
    }

    selected_loss = losses[loss]()

    if optimizer == 'sgd':
        selected_optimizer = partial(torch.optim.SGD, lr=learning_rate, momentum=0.9, weight_decay=0.001, nesterov=True)
    elif optimizer == 'adam':
        selected_optimizer = partial(torch.optim.Adam, lr=learning_rate)
    else:
        raise RuntimeError()

    training_ds = CartoonLSTMDataset(
        file_path=get_subset(dataset_path=source, subset=Subset.TRAINING),
    )
    validation_ds = CartoonLSTMDataset(
        file_path=get_subset(dataset_path=source, subset=Subset.VALIDATION),
    )

    train_lstm_model(
        criterion=selected_loss,
        optimizer=selected_optimizer,
        scheduler=partial(lr_scheduler.StepLR, step_size=8, gamma=0.1),
        training_dataset=training_ds,
        validation_dataset=validation_ds,
        batch_size=batch_size,
        device=device,
        num_epochs=epochs,
    )


def setup_train_lstm(parser: argparse.ArgumentParser, group):
    group.add_argument('--train_lstm', action="store_true")

    def train(args, device):
        if not args.train_lstm:
            return

        pipeline(
            source=args.source,
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            loss=args.loss,
            optimizer=args.optimizer,
            device=device,
        )

    return train
