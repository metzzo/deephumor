from django.core.management.base import BaseCommand, CommandError

from annotator.models import Cartoon
from image_preprocessor.core.find_duplicates import find_duplicates


class Command(BaseCommand):
    help = 'Filters duplicate cartoons'

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        duplicates = find_duplicates('./media/photos/')
        for _, duplicates in duplicates.items():
            is_first = True
            for entry in duplicates:
                entry_id = int(entry.split('/')[-2])
                cartoon = Cartoon.objects.get(id=entry_id)
                if is_first:
                    is_first = False
                    cartoon.annotated = False
                else:
                    cartoon.annotated = True
                    cartoon.relevant = False
                cartoon.save()
            print(duplicates)
