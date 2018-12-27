import csv

from django.core.management.base import BaseCommand, CommandError

from annotator.models import Cartoon, relevant_cartoon_queryset, FunninessAnnotation
from annotator.tasks import import_comic

import os
import pandas as pd
import shutil
import pickle

EXPORT_DIR = 'export/'

class Command(BaseCommand):
    help = 'Exports the Dataset. Either as CSV or Pickle file'

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
        pickle.dump(df, open(os.path.join(EXPORT_DIR, 'export.p'), "wb"))