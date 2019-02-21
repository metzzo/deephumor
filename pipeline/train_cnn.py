import argparse
from functools import partial

import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, L1Loss
from torch.nn.functional import softmax

from architectures.pretrained_cnn import PretrainedCNNCartoonModel
from datamanagement.dataset import get_subset
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


def pipeline(source, model, epochs, batch_size, loss, device):
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
        'l1': partial(L1Loss, reduction='mean'),
        'custom': custom_loss
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
            device=device,
        )

    return train
