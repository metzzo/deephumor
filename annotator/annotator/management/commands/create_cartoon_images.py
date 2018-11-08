from django.core.management.base import BaseCommand, CommandError
from PIL import Image, ImageOps, ImageStat
from annotator.models import Cartoon
from annotator.tasks import import_comic

import os
from pathlib import Path
from PIL import Image, ImageOps, ImageStat


class Command(BaseCommand):
    help = 'Creates the cartoon images from the original images if specified'


    def handle(self, *args, **options):
        cartoons = list(Cartoon.objects.all())

        for c in cartoons:
            if c.custom_dimensions is not None and c.custom_dimensions != '0 0 0 0' and len(c.custom_dimensions) > 0:
                try:
                    im = Image.open('./annotator/media/photos/{0}/original_cartoon.jpg'.format(c.id))
                    dims = list(map(lambda x: int(x), c.custom_dimensions.split(' ')))
                    dims[2] += dims[0]
                    dims[3] += dims[1]
                    print(c.id)
                    im = im.crop(dims)
                    im.save('./annotator/media/photos/{0}/cartoon.jpg'.format(c.id), "JPEG")
                except IOError:
                    print("cannot create thumbnail")

