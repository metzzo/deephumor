import argparse
from functools import partial

import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, L1Loss, NLLLoss
from torch.nn.functional import softmax

from cnn_experiments.factory import get_model
from datamanagement.factory import get_subset
from models.train_cnn import train_cnn_model
from torch.optim import lr_scheduler


def custom_loss():
    def calc(outputs, label):
        outputs = softmax(outputs)
        arr = [0] * len(outputs)
        for j in range(0, len(outputs)):
            c = label[j]
            for i in range(1, 8):
                arr[j] += outputs[j][i - 1] * (i - c) * (i - c) if i != c else 0
        return torch.sum(Variable(torch.tensor(arr, dtype=torch.float32), requires_grad=True))

    return calc


def pipeline(source, model, epochs, batch_size, learning_rate, loss, optimizer, device):
    from datamanagement.subset import Subset

    losses = {
        'cel': CrossEntropyLoss,
        'l1': partial(L1Loss, reduction='mean'),
        'nll': NLLLoss,
        'custom': custom_loss
    }
    selected_loss = losses[loss]()
    selected_model = get_model(model_name=model)

    if optimizer == 'sgd':
        selected_optimizer = partial(torch.optim.SGD, lr=learning_rate, momentum=0.9, weight_decay=0.001, nesterov=True)
    elif optimizer == 'adam':
        selected_optimizer = partial(torch.optim.Adam, lr=learning_rate)
    else:
        raise RuntimeError()

    training_ds = selected_model.Dataset(
        file_path=get_subset(dataset_path=source, subset=Subset.TRAINING),
        model=selected_model,
        trafo=selected_model.get_train_transformation(),
    )
    validation_ds = selected_model.Dataset(
        file_path=get_subset(dataset_path=source, subset=Subset.VALIDATION),
        model=selected_model,
        trafo=selected_model.get_validation_transformation(),
    )

    train_cnn_model(
        model=selected_model,
        criterion=selected_loss,
        optimizer=selected_optimizer,
        scheduler=partial(lr_scheduler.StepLR, step_size=10, gamma=0.1),
        training_dataset=training_ds,
        validation_dataset=validation_ds,
        batch_size=batch_size,
        device=device,
        num_epochs=epochs,
    )


def setup_train_cnn(parser: argparse.ArgumentParser, group):
    group.add_argument('--train_cnn', action="store_true")

    def train(args, device):
        if not args.train_cnn:
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
