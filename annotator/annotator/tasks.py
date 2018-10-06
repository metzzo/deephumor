from __future__ import absolute_import, unicode_literals
from celery import shared_task
from django.core.files.base import ContentFile

from annotator.models import Cartoon
from image_preprocessor.core.preprocess_image import preprocess_image
from image_preprocessor.core.utility import to_jpeg

@shared_task
def import_comic(file):
    print('Importing %s' % file)

    img, punchline, original_img = preprocess_image(path=file)

    cartoon = Cartoon()
    cartoon.punchline = punchline
    cartoon.name = file
    cartoon.save()
    cartoon.img.save('cartoon.jpg', ContentFile(to_jpeg(img)), save=True)
    cartoon.original_img.save('original_cartoon.jpg', ContentFile(to_jpeg(original_img)), save=True)
    cartoon.save()
