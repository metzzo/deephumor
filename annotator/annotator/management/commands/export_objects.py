import csv

from PIL import Image
from django.core.management.base import BaseCommand, CommandError

from ...models import Cartoon, relevant_cartoon_queryset, FunninessAnnotation, ImageAnnotationCollection
from ...tasks import import_comic
from random import shuffle

import os
import pandas as pd
import shutil
import pickle

EXPORT_DIR = '../export/objects/'

class Command(BaseCommand):
    help = 'Exports the image annotations'

    def save_records(self, records, name):
        df = pd.DataFrame.from_records(data=records, columns=(
            'filename',
            'annotation_id',
            'cartoon_id',
            'cl',
        ))
        df.to_csv(
            os.path.join(EXPORT_DIR, '{0}_set.csv'.format(name)),
            sep=';',
            encoding='utf-8',
            index=False,
            quoting=csv.QUOTE_NONNUMERIC
        )
        pickle.dump(df, open(os.path.join(EXPORT_DIR, '{0}_set.p'.format(name)), "wb"))

    def handle(self, *args, **options):
        if os.path.exists(EXPORT_DIR) and os.path.isdir(EXPORT_DIR):
            shutil.rmtree(EXPORT_DIR)

        os.mkdir(EXPORT_DIR)

        records = []
        for collection in ImageAnnotationCollection.objects.all().filter(annotated=True):
            for annotation in collection.imageannotation_set.all():
                # cut out image
                filename = 'cartoon_{0}_object_{1}.jpg'.format(collection.cartoon.id, annotation.id)
                try:
                    im = Image.open('./media/photos/{0}/cartoon.jpg'.format(collection.cartoon.id))
                    dims = list(map(lambda x: int(x), annotation.dimensions.split(' ')))
                    dims[2] += dims[0]
                    dims[3] += dims[1]
                    im = im.crop(dims)
                    im.save(os.path.join(EXPORT_DIR, filename), "JPEG")
                except IOError:
                    print("cannot create thumbnail")

                fields = [
                    filename,
                    annotation.id,
                    collection.cartoon.id,
                    annotation.annotation_class.name
                ]
                records += [fields]
        shuffle(records)

        train_boundary = int(len(records)*0.6)
        validation_boundary = int(len(records)*0.3)

        self.save_records(records=records[:train_boundary], name='train')
        self.save_records(records=records[train_boundary:train_boundary + validation_boundary], name='validation')
        self.save_records(records=records[train_boundary + validation_boundary:], name='test')
