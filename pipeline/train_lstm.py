import argparse
from functools import partial

import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, L1Loss
from torch.nn.functional import softmax

from datamanagement.factory import get_subset
from models.train_cnn import train_cnn_model
from torch.optim import lr_scheduler


def pipeline(source, model, epochs, batch_size, loss, device):
    from datamanagement.subset import Subset
    from datamanagement.cartoon_dataset import CartoonDataset

    torch.manual_seed(42)

    models = [

    ]
    losses = {

    }

    selected_model = next((x for x in models if x.__name__ == model), None)()
    selected_loss = losses[loss]()

    training_ds = CartoonDataset(file_path=get_subset(dataset_path=source, subset=Subset.TRAINING), model=selected_model)
    validation_ds = CartoonDataset(file_path=get_subset(dataset_path=source, subset=Subset.VALIDATION), model=selected_model)


    train_cnn_model(
        model=selected_model,
        criterion=selected_loss,
        optimizer=partial(torch.optim.SGD, lr=0.001, momentum=0.9, weight_decay=0.00001, nesterov=True),
        scheduler=partial(lr_scheduler.StepLR, step_size=8, gamma=0.1),
        training_dataset=training_ds,
        validation_dataset=validation_ds,
        batch_size=batch_size,
        device=device,
        num_epochs=epochs,
    )


def setup_train_cnn(parser: argparse.ArgumentParser, group):
    group.add_argument('--train_cnn', action="store_true")

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--model', type=str)
    parser.add_argument('--loss', type=str)

    def train(args, device):
        if not args.train_cnn:
            return

        pipeline(
            source=args.source,
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            loss=args.loss,
            device=device
        )

    return train
