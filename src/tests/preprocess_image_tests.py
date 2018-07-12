from hamcrest import *

from core.preprocess_image import preprocess_image


def test_sampleimage1():
    # arrange
    path = './resources/sample_comic.jpg'

    # act
    cartoon, punchline = preprocess_image(path=path)

    # assert
    assert_that(punchline, is_('When dumb animals attempt murder.'))


def test_sampleimage2():
    # arrange
    path = './resources/sample_comic2.jpg'

    # act
    cartoon, punchline = preprocess_image(path=path)

    # assert
    # Something\'s wrong here, Harriet...This is starting to\nlook less and less like Interstate 95.
    assert_that(punchline, is_('”Something’s wrong here. Hamet . . . This Is stoning to\nlook less and less like Interstate 95."'))


def test_wrong_orientation():
    # arrange
    path = './resources/wrong_orientation.jpg'

    # act
    cartoon, punchline = preprocess_image(path=path)

    # assert
    assert_that(punchline, is_('"And so I’ve reached the conctusion. gentlemen. that\ntho Wonket Wiener Company Is riddled\nwith Incompetence."'))
