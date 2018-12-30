from pipeline.datamanagement.batches import BatchGenerator
from pipeline.datamanagement.dataset import CartoonDataset
from pipeline.processing.ops import normalize_size
from pipeline.settings import DATASET_PATH


def test_batches():
    bg = BatchGenerator(
        dataset=CartoonDataset(
            file_path=DATASET_PATH
        ),
        num=150,
        shuffle=False,
        op=normalize_size(target_width=128, target_height=128)
    )
    assert bg is not None
