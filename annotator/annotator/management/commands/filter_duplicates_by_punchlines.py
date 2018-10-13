from django.core.management.base import BaseCommand, CommandError

from annotator.models import Cartoon
from image_preprocessor.core.find_duplicates import find_duplicates


class Command(BaseCommand):
    help = 'Filters duplicate cartoons by punchlines'

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        duplicates = 0
        objects = list(Cartoon.objects.all())
        progress = 0
        for c1 in objects:
            progress += 1
            print("{0} duplicates at {1} cartoon".format(duplicates, progress))
            for c2 in objects:
                if c1.pk != c2.pk and c1.punchline == c2.punchline and c1.punchline != 'unknown' and len(c1.punchline) > 1:
                    if c1.pk < c2.pk:
                        c2.annotated = True
                        c2.relevant = False
                        c2.duplicate_of = c1
                        c2.save()
                        duplicates += 1
