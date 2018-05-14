from hamcrest import *

from core.preprocess_image import preprocess_image


def test_sampleimage(path):
    # arrange
    path = './resources/sample_comic.jpg'

    # act
    result = preprocess_image(path=path)

    # assert
    assert_that(result, is_not(None))
