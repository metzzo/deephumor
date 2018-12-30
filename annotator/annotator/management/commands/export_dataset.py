import csv

from django.core.management.base import BaseCommand, CommandError

from ...models import Cartoon, relevant_cartoon_queryset, FunninessAnnotation
from ...tasks import import_comic
from random import shuffle

import os
import pandas as pd
import shutil
import pickle

EXPORT_DIR = 'export/'

class Command(BaseCommand):
    help = 'Exports the Dataset. Either as CSV or Pickle file'

    def save_records(self, records, name):
        df = pd.DataFrame.from_records(data=records, columns=(
            'filename',
            'punchline',
            'funniness',
            'i_understand'
        ))
        df.to_csv(
            os.path.join(EXPORT_DIR, 'data.csv'),
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
        for cartoon in relevant_cartoon_queryset():
            funniness_annotation = FunninessAnnotation.objects.all().filter(cartoon_id=cartoon.id).first()

            filename = os.path.basename(cartoon.name)

            fields = [
                filename,
                cartoon.punchline.replace('\n', '\\n').replace('\r', '').replace('"', "'"),
                funniness_annotation.funniness,
                funniness_annotation.i_understand,
            ]

            source_dir = os.path.join(
                os.getcwd(),
                'annotator/',
                '.' + cartoon.img.url
            )

            shutil.copy(source_dir, os.path.join(
                EXPORT_DIR,
                filename)
            )

            records += [fields]
        shuffle(records)

        train_boundary = int(len(records)*0.6)
        validation_boundary = int(len(records)*0.3)

        self.save_records(records=records[:train_boundary], name='train')
        self.save_records(records=records[train_boundary:train_boundary + validation_boundary], name='validation')
        self.save_records(records=records[train_boundary + validation_boundary:], name='test')
