import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader

from datamanagement.dataset import get_subset
from evaluation.overall_evaluation import OverallEvaluation
from settings import WEIGHT_DECAY, BATCH_SIZE

CartoonDataLoader = partial(DataLoader, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


def pipeline(source, epochs=1):
    from architectures.cnn import SimpleCNNCartoonModel
    from datamanagement.subset import Subset
    from models.cnn_model import CnnClassifier
    from evaluation.accuracy_evaluation import AccuracyEvaluation

    use_cuda = torch.cuda.is_available()
    print("Uses CUDA: {0}".format(use_cuda))
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.empty_cache()

    from datamanagement.dataset import CartoonDataset
    training_ds = CartoonDataset(file_path=get_subset(dataset_path=source, subset=Subset.TRAINING))
    validation_ds = CartoonDataset(file_path=get_subset(dataset_path=source, subset=Subset.VALIDATION))
    training_dl = CartoonDataLoader(dataset=training_ds)
    validation_dl = DataLoader(dataset=validation_ds, batch_size=BATCH_SIZE)

    net = SimpleCNNCartoonModel()
    net.to(device)
    clf = CnnClassifier(
        net=net,
        input_shape=(3, 32, 32),
        num_classes=7,
        lr=0.001,
        wd=WEIGHT_DECAY,
    )

    # training
    print('Training')

    for i in range(epochs):
        print('{0} / {1}'.format(i + 1, epochs))
        training_evaluation = OverallEvaluation(num=len(training_dl), ignore_loss=False)
        for samples in training_dl:
            _, batch_images, _, batch_funniness = samples
            batch_images, batch_funniness = batch_images.to(device), batch_funniness.to(device)
            loss = clf.train(
                data=batch_images,
                labels=batch_funniness
            )
            predictions = clf.predict(data=batch_images)
            training_evaluation.add_entry(predictions=predictions, actual_label=batch_funniness, loss=loss)

        print("Training Evaluation:")
        print(training_evaluation)

        validation_evaluation = OverallEvaluation(num=len(validation_dl), ignore_loss=True)
        with torch.set_grad_enabled(False):
            for samples in validation_dl:
                _, batch_images, _, batch_funniness = samples
                batch_images = batch_images.to(device)
                batch_funniness = batch_funniness.to(device)
                predictions = clf.predict(data=batch_images)
                validation_evaluation.add_entry(predictions=predictions, actual_label=batch_funniness, loss=0)
            print("Validation Evaluation:")
            print(validation_evaluation)


def setup_train(parser: argparse.ArgumentParser, group):
    group.add_argument('--train', action="store_true")

    parser.add_argument('--epochs', type=int)

    def train(args):
        if not args.train:
            return

        pipeline(
            source=args.source,
            epochs=args.epochs,
        )

    return train
