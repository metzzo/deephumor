import argparse
from functools import partial

import torch
from torch.nn import CrossEntropyLoss, L1Loss

from architectures.pretrained_cnn import PretrainedCNNCartoonModel
from datamanagement.dataset import get_subset
from models.train_cnn import train_cnn_model
from torch.optim import lr_scheduler

def pipeline(source, model, epochs, batch_size, loss):
    from datamanagement.subset import Subset
    from datamanagement.dataset import CartoonDataset

    from architectures.simple_regression_cnn import SimpleRegressionCNNCartoonModel
    from architectures.simple_classification_cnn import SimpleClassificationCNNCartoonModel

    models = [
        SimpleRegressionCNNCartoonModel,
        SimpleClassificationCNNCartoonModel,
        PretrainedCNNCartoonModel,
    ]
    losses = {
        'cel': CrossEntropyLoss,
        'l1': partial(L1Loss, reduction='mean')
    }

    use_cuda = torch.cuda.is_available()
    print("Uses CUDA: {0}".format(use_cuda))
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.empty_cache()

    training_ds = CartoonDataset(file_path=get_subset(dataset_path=source, subset=Subset.TRAINING))
    validation_ds = CartoonDataset(file_path=get_subset(dataset_path=source, subset=Subset.VALIDATION))

    selected_model = next((x for x in models if x.__name__ == model), None)

    train_cnn_model(
        model=selected_model(),
        criterion=losses[loss](),
        optimizer=partial(torch.optim.SGD, lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True),
        scheduler=partial(lr_scheduler.StepLR, step_size=8, gamma=0.1),
        training_dataset=training_ds,
        validation_dataset=validation_ds,
        batch_size=batch_size,
        device=device,
        num_epochs=epochs,
    )


def setup_train(parser: argparse.ArgumentParser, group):
    group.add_argument('--train', action="store_true")

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--model', type=str)
    parser.add_argument('--loss', type=str)

    def train(args):
        if not args.train:
            return

        pipeline(
            source=args.source,
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            loss=args.loss
        )

    return train
