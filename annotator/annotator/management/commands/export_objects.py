import csv

from PIL import Image
from django.core.management.base import BaseCommand, CommandError

from ...models import Cartoon, relevant_cartoon_queryset, FunninessAnnotation, ImageAnnotationCollection, \
    ImageAnnotationClass
from ...tasks import import_comic
from random import shuffle

import os
import pandas as pd
import shutil
import pickle
import random

EXPORT_DIR = '../export/objects/'


def overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    return not (x1+w1<x2 or x2+w2<x1 or y1+h1<y2 or y2+h2<y1)


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
        for collection in ImageAnnotationCollection.objects.all(): #.filter(annotated=True):
            samples = []
            try:
                im = Image.open('./media/photos/{0}/cartoon.jpg'.format(collection.cartoon.id))
                for annotation in collection.imageannotation_set.all():
                    # cut out image
                    filename = 'cartoon_{0}_object_{1}.jpg'.format(collection.cartoon.id, annotation.id)

                    dims = list(map(lambda x: int(x), annotation.dimensions.split(' ')))
                    samples.append(dims)
                    dims[2] += dims[0]
                    dims[3] += dims[1]
                    copy = im.copy()
                    copy = copy.crop(dims)
                    copy.save(os.path.join(EXPORT_DIR, filename), "JPEG")
                    print("Save {0}".format(collection.id))
                    fields = [
                        filename,
                        annotation.id,
                        collection.cartoon.id,
                        annotation.annotation_class.name
                    ]
                    records += [fields]
                for idx, sample in enumerate(samples):
                    for i in range(1,100):
                        w, h = random.randint(50, 300), random.randint(50, 300)
                        bg_frame = [
                            random.randint(0, max(0, im.width - w)),
                            random.randint(0, max(0, im.height - h)),
                            w,
                            h
                        ]
                        not_overlap = True
                        for sample2 in samples:
                            if overlap(*bg_frame, *sample2):
                                not_overlap = False
                                break
                        if not_overlap:
                            filename = 'cartoon_{0}_background_{1}.jpg'.format(collection.cartoon.id, idx)
                            bg_frame[2] += bg_frame[0]
                            bg_frame[3] += bg_frame[1]
                            copy = im.copy()
                            copy = copy.crop(bg_frame)
                            copy.save(os.path.join(EXPORT_DIR, filename), "JPEG")
                            fields = [
                                filename,
                                -1,
                                collection.cartoon.id,
                                'BACKGROUND'
                            ]
                            records += [fields]
                            break


            except IOError:
                print("cannot load cartoon")
        shuffle(records)

        train_boundary = 0 #int(len(records)*0.6)
        validation_boundary = len(records) #int(len(records)*0.3)

        self.save_records(records=records[:train_boundary], name='train')
        self.save_records(records=records[train_boundary:train_boundary + validation_boundary], name='validation')
        self.save_records(records=records[train_boundary + validation_boundary:], name='test')

        # extract classes
        classes = []
        for cl in ImageAnnotationClass.objects.all():
            classes.append(cl.name)

        pickle.dump(classes, open(os.path.join(EXPORT_DIR, 'classes.p'), "wb"))
