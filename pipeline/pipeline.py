from functools import partial

import torch
from torch.utils.data import DataLoader
from settings import WEIGHT_DECAY, BATCH_SIZE

CartoonDataLoader = partial(DataLoader, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


def iterate_over(dataloader: DataLoader, device: torch.device, epochs: int):
    for i in range(epochs):
        print('{0} / {1}'.format(i + 1, epochs))
        for samples in dataloader:
            _, batch_images, _, batch_funniness = samples
            print("yolo")
            yield batch_images.to(device), batch_funniness.to(device)


def pipeline(epochs=1):
    from architectures.cnn import SimpleCNNCartoonModel
    from datamanagement.subset import Subset
    from evaluation.Evaluation import Evaluation
    from models.cnn_model import CnnClassifier

    use_cuda = torch.cuda.is_available()
    print("Uses CUDA: {0}".format(use_cuda))
    device = torch.device("cuda:0" if use_cuda else "cpu")

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
    training_iterator = partial(iterator, dataloader=training_dl)
    validation_iterator = partial(iterator, dataloader=validation_dl)

    net = SimpleCNNCartoonModel()
    clf = CnnClassifier(
        net=net,
        input_shape=(3, 32, 32),
        num_classes=7,
        lr=0.01,
        wd=WEIGHT_DECAY,
    )

    # training
    print('Training Phase')
    evaluation = Evaluation(num=BATCH_SIZE)
    for batch_images, batch_funniness in training_iterator():
        loss = clf.train(
            data=batch_images,
            labels=batch_funniness
        )
        pred = clf.predict(data=batch_images)
        evaluation.add_entry(predicted_label=pred, actual_label=batch_funniness, loss=loss)

    # validation
    print('Test Phase')
    with torch.set_grad_enabled(False):
        for batch_images, batch_funniness in validation_iterator():
            clf.predict(data=batch_images)


if __name__ == '__main__':
    pipeline()
