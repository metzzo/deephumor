import csv
import io

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand

from ...models import relevant_cartoon_queryset, FunninessAnnotation
from random import shuffle

import os
import pandas as pd
import shutil
import pickle

EXPORT_DIR = 'export/'

class Command(BaseCommand):
    help = 'Exports the Dataset. Either as CSV or Pickle file'

    def add_arguments(self, parser):
        parser.add_argument('annotator', type=str)

    def save_records(self, records, name):
        TO_PRINT = True

        df = pd.DataFrame.from_records(data=records, columns=(
            'filename',
            'punchline',
            'funniness',
            'i_understand',
            'id'
        ))

        target = io.StringIO() if TO_PRINT else os.path.join(EXPORT_DIR, 'data.csv');

        df.to_csv(
            target,
            sep=';',
            encoding='utf-8',
            index=False,
            quoting=csv.QUOTE_NONNUMERIC
        )

        if TO_PRINT:
            print(target.getvalue())
        else:
            pickle.dump(df, open(os.path.join(EXPORT_DIR, '{0}_set.p'.format(name)), "wb"))

    def handle(self, *args, **options):
        DO_COPY = False
        DO_SPLIT = False

        if os.path.exists(EXPORT_DIR) and os.path.isdir(EXPORT_DIR):
            shutil.rmtree(EXPORT_DIR)

        os.mkdir(EXPORT_DIR)

        user = User.objects.all().filter(username=options['annotator']).first()
        records = []
        for cartoon in relevant_cartoon_queryset():
            funniness_annotation = FunninessAnnotation.objects.all().filter(cartoon_id=cartoon.id).filter(annotated_by=user).first()
            if funniness_annotation:
                filename = os.path.basename(cartoon.name)

                fields = [
                    filename,
                    cartoon.punchline.replace('\n', '\\n').replace('\r', '').replace('"', "'"),
                    funniness_annotation.funniness,
                    funniness_annotation.i_understand,
                    funniness_annotation.id,
                ]
                if DO_COPY:
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

        if DO_SPLIT:
            train_boundary = int(len(records)*0.6)
            validation_boundary = int(len(records)*0.3)

            self.save_records(records=records[:train_boundary], name='train')
            self.save_records(records=records[train_boundary:train_boundary + validation_boundary], name='validation')
            self.save_records(records=records[train_boundary + validation_boundary:], name='test')
        else:
            self.save_records(records=records, name='dataset')
