from django.core.management.base import BaseCommand, CommandError

from annotator.models import Cartoon
from image_preprocessor.core.find_duplicates import find_duplicates


class Command(BaseCommand):
    help = 'Filters duplicate cartoons by images'

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        duplicates = find_duplicates('./annotator/media/photos/')
        for _, duplicates in duplicates.items():
            is_first = True
            ids = map(lambda entry: int(entry.split('/')[-2]), duplicates)
            ids = sorted(ids)
            try:
                original = Cartoon.objects.get(id=ids[0])
            except:
                continue
            for entry_id in ids:
                try:
                    cartoon = Cartoon.objects.get(id=entry_id)
                except:
                    continue

                if not is_first:
                    cartoon.annotated = True
                    cartoon.relevant = False

                if original.id != entry_id:
                    cartoon.duplicate_of = original
                is_first = False
                cartoon.save()
            print(ids)
