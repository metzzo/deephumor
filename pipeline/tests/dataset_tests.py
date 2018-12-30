from pipeline.datamanagement.dataset import CartoonDataset
from pipeline.settings import DATASET_PATH


def test_dataset():
    ds = CartoonDataset(
        file_path=DATASET_PATH
    )

    assert ds is not None
    assert len(ds) == 2487
    assert ds[0].funniness == 2
    assert len(ds[0].punchline) > 10
    assert ds[2486].funniness == 1
