from pipeline.deephumor.batches import BatchGenerator
from pipeline.deephumor.dataset import CartoonDataset
from pipeline.settings import DATASET_PATH


def test_batches():
    bg = BatchGenerator(
        dataset=CartoonDataset(
            file_path=DATASET_PATH
        ),
        num=150,
        shuffle=False
    )
    assert bg is not None
