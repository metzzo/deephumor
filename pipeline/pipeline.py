from datamanagement.dataset import CartoonDataset
from pipeline.models.cnn_model import train


def pipeline():
    training_ds = CartoonDataset(
        fdir="../../cifar-10",
        subset=Subset.TRAINING
    )
    validation_ds = CartoonDataset(
        fdir="../../cifar-10",
        subset=Subset.VALIDATION
    )
    test_ds = CartoonDataset(
        fdir="../../cifar-10",
        subset=Subset.TEST
    )


if __name__ == '__main__':
    pipeline()
