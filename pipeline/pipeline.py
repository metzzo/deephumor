from functools import partial

import torch
from torch.utils.data import DataLoader
from settings import WEIGHT_DECAY, BATCH_SIZE

CartoonDataLoader = partial(DataLoader, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


def iterate_over(dataloader: DataLoader, device: torch.device, epochs: int, evaluation):
    if evaluation:
        evaluation.reset()

    for i in range(epochs):
        print('{0} / {1}'.format(i + 1, epochs))
        for samples in dataloader:
            _, batch_images, _, batch_funniness = samples
            yield batch_images.to(device), batch_funniness.to(device)
        if evaluation:
            print(evaluation)
            evaluation.reset()


def pipeline(epochs=1):
    from architectures.cnn import SimpleCNNCartoonModel
    from datamanagement.subset import Subset
    from evaluation.evaluation import Evaluation
    from models.cnn_model import CnnClassifier

    use_cuda = torch.cuda.is_available()
    print("Uses CUDA: {0}".format(use_cuda))
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.empty_cache()

    from datamanagement.dataset import CartoonDataset
    training_ds = CartoonDataset(
        subset=Subset.TRAINING
    )
    validation_ds = CartoonDataset(
        subset=Subset.VALIDATION
    )

    training_dl = CartoonDataLoader(dataset=training_ds)
    validation_dl = DataLoader(dataset=validation_ds)

    iterator = partial(iterate_over, epochs=epochs, device=device)

    net = SimpleCNNCartoonModel()
    net.to(device)
    clf = CnnClassifier(
        net=net,
        input_shape=(3, 32, 32),
        num_classes=7,
        lr=0.01,
        wd=WEIGHT_DECAY,
    )

    # training
    print('Training Phase')

    evaluation = Evaluation(num=len(training_dl), batch_size=BATCH_SIZE)
    training_iterator = iterator(dataloader=training_dl, evaluation=evaluation)
    for batch_images, batch_funniness in training_iterator:
        loss = clf.train(
            data=batch_images,
            labels=batch_funniness
        )
        predictions = clf.predict(data=batch_images)
        evaluation.add_entry(predictions=predictions, actual_label=batch_funniness, loss=loss)

    # validation
    print('Test Phase')

    evaluation = Evaluation(num=1, batch_size=len(validation_ds), ignore_loss=True)
    validation_iterator = iterator(dataloader=validation_dl, evaluation=evaluation)
    with torch.set_grad_enabled(False):
        for batch_images, batch_funniness in validation_iterator:
            predictions = clf.predict(data=batch_images)
            evaluation.add_entry(predictions=predictions, actual_label=batch_funniness)


if __name__ == '__main__':
    pipeline()
